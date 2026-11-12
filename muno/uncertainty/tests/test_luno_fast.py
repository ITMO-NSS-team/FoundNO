from __future__ import annotations

import torch

from luno_torch.affine import LastBlockSuffix
from luno_torch.ggn import GGNMatvec, LiftingGGNMatvec, randomized_eigh, to_dense
from luno_torch.jacobian import LastFNOBlockWeightJacobian
from luno_torch.lino_ops import LinearOperator

from .conftest import close, dense_ggn, dense_jacobian


def _count_core_forwards(core, fn):
    """Сколько раз ``fn`` вызвал forward ядра целиком."""
    calls = [0]
    h = core.register_forward_hook(lambda *a: calls.__setitem__(0, calls[0] + 1))
    try:
        fn()
    finally:
        h.remove()
    return calls[0]


def test_ggn_matvec_matches_dense_last_block(tiny):
    xs = tiny.xs(3)
    G = dense_ggn(tiny.model_fn, xs, tiny.w0)
    mv = GGNMatvec(tiny.model_fn, tiny.w0, xs, fwd_batch=2)
    V = torch.randn(tiny.w0.numel(), 5, dtype=tiny.w0.dtype)
    assert close(mv._matmul(V), G @ V)
    assert close(mv._matmul(V[:, 0]), G @ V[:, 0])


def test_ggn_matvec_matches_dense_lifting(tiny):
    xs = tiny.xs(3)
    lw = tiny.lift_wrapper
    G = dense_ggn(lw.model_fn, xs, lw.w0)
    mv = LiftingGGNMatvec(lw.model_fn, lw.w0, xs, fwd_batch=3)
    assert close(to_dense(mv), G)


def test_ggn_matvec_one_graph_per_sample(tiny):
    xs = tiny.xs(2)
    k, fwd_batch = 6, 2
    mv = GGNMatvec(tiny.model_fn, tiny.w0, xs, fwd_batch=fwd_batch)
    V = torch.randn(tiny.w0.numel(), k, dtype=tiny.w0.dtype)
    n = _count_core_forwards(tiny.core, lambda: mv._matmul(V))
    # На сэмпл: 1 forward под обратный граф + по одному primal на чанк jvp.
    assert n == len(xs) * (1 + k // fwd_batch)


def test_suffix_reproduces_model_and_jacobian(tiny):
    x = tiny.xs(1)[0]
    suffix = LastBlockSuffix(tiny.affine, tiny.model_fn, tiny.w0)
    fn, state = suffix.bind(x)
    w = tiny.w0 + 0.1 * torch.randn_like(tiny.w0)
    assert close(fn(state, w), tiny.model_fn(x, w))

    J = dense_jacobian(tiny.model_fn, x, tiny.w0)
    jac = LastFNOBlockWeightJacobian(fn, state, tiny.w0, fwd_batch=3)
    V = torch.randn(tiny.w0.numel(), 4, dtype=tiny.w0.dtype)
    U = torch.randn(J.shape[0], 4, dtype=tiny.w0.dtype)
    assert close(jac._matmul(V), J @ V)
    assert close(jac.T._matmul(U), J.T @ U)


def test_ggn_matvec_with_suffix_bind(tiny):
    xs = tiny.xs(3)
    G = dense_ggn(tiny.model_fn, xs, tiny.w0)
    suffix = LastBlockSuffix(tiny.affine, tiny.model_fn, tiny.w0)
    mv = GGNMatvec(tiny.model_fn, tiny.w0, xs, fwd_batch=2, bind=suffix.bind)
    V = torch.randn(tiny.w0.numel(), 6, dtype=tiny.w0.dtype)
    assert close(mv._matmul(V), G @ V)
    # Префикс считается один раз на сэмпл, сколько бы ни было колонок.
    n = _count_core_forwards(tiny.core, lambda: mv._matmul(V))
    assert n == len(xs)


class _DenseOp(LinearOperator):
    def __init__(self, A):
        self.A = A
        self.calls = 0

    def _matmul(self, x):
        self.calls += 1
        return self.A @ x

    def shape(self):
        return tuple(self.A.shape)

    @property
    def dtype(self):
        return self.A.dtype

    @property
    def device(self):
        return self.A.device


def _subspace_overlap(U, U_ref):
    """Наименьший косинус главных углов между span(U) и span(U_ref)."""
    return torch.linalg.svdvals(U_ref.T @ U).min().item()


def test_nystrom_eigh_matches_dense(tiny):
    G = dense_ggn(tiny.model_fn, tiny.xs(3), tiny.w0).detach()
    evals, evecs = torch.linalg.eigh(G)
    evals, evecs = evals.flip(0), evecs.flip(1)
    rank = 8
    for power_iter in (1, 2):
        op = _DenseOp(G)
        U, S = randomized_eigh(op, n=G.shape[0], rank=rank, oversample=10,
                               power_iter=power_iter, seed=0)
        assert op.calls == power_iter + 1
        assert U.shape == (G.shape[0], rank)
        assert close(U.T @ U, torch.eye(rank, dtype=G.dtype), tol=1e-6)
        assert close(S, evals[:rank], tol=2e-2 if power_iter == 1 else 2e-3)
        # Подпространство сравниваем там, где в спектре есть зазор (после 2-й и
        # 3-й пары); дальше хвост почти вырожден и базис определён неоднозначно.
        assert _subspace_overlap(U[:, :2], evecs[:, :2]) > 0.99999
        assert _subspace_overlap(U[:, :3], evecs[:, :3]) > (0.99 if power_iter == 1 else 0.9999)


def test_nystrom_eigh_rank_deficient():
    torch.manual_seed(0)
    X = torch.randn(40, 3, dtype=torch.float64)
    A = X @ X.T
    U, S = randomized_eigh(_DenseOp(A), n=40, rank=6, oversample=4, power_iter=1)
    assert U.shape == (40, 6) and torch.isfinite(U).all()
    assert close(S[:3], torch.linalg.eigvalsh(A).flip(0)[:3], tol=1e-8)
    assert S[3:].abs().max() < 1e-8 * S[0]
    assert close((U * S) @ U.T, A, tol=1e-8)


def _posterior_cov(tiny, xs, rank=6, prior_prec=0.5):
    from luno_torch.factory import create_luno_cov
    from luno_torch.ggn import LowRankTerms

    mv = GGNMatvec(tiny.model_fn, tiny.w0, xs, fwd_batch=4)
    U, S = randomized_eigh(mv, n=tiny.w0.numel(), rank=rank, oversample=6)
    return create_luno_cov(LowRankTerms(U=U, S=S), {"prior_prec": prior_prec})


def test_hutchinson_fast_path_equals_probe_form(tiny):
    """Sigma^{1/2} в пространстве весов == раздельные пробы (A, B, W) при том же Z."""
    from luno_torch.jacobian import (
        _var_of_congruence_hutchinson,
        sample_congruence_probes,
        var_from_congruence_probes,
    )

    xs = tiny.xs(3)
    cov = _posterior_cov(tiny, xs)
    x = tiny.xs(1, seed=7)[0]
    n_probe = 6
    jac = LastFNOBlockWeightJacobian(tiny.model_fn, x, tiny.w0, fwd_batch=4)

    torch.manual_seed(123)
    fast = _var_of_congruence_hutchinson(jac, cov, n_probe)
    torch.manual_seed(123)
    A, B, W = sample_congruence_probes(jac, cov.U, n_probe)
    ref = var_from_congruence_probes(A, B, W, cov)
    assert close(fast, ref, tol=1e-6)

    # И J@U больше не считается: только n_probe проходов (чанками по fwd_batch).
    n = _count_core_forwards(
        tiny.core, lambda: _var_of_congruence_hutchinson(jac, cov, n_probe)
    )
    assert n == -(-n_probe // 4)


def test_hutchinson_converges_to_exact_variance(tiny):
    from luno_torch.jacobian import _var_of_congruence_hutchinson

    xs = tiny.xs(3)
    cov = _posterior_cov(tiny, xs)
    x = tiny.xs(1, seed=7)[0]
    J = dense_jacobian(tiny.model_fn, x, tiny.w0).detach()
    exact = ((J @ cov.todense()) * J).sum(dim=1)

    suffix = LastBlockSuffix(tiny.affine, tiny.model_fn, tiny.w0)
    fn, state = suffix.bind(x)
    jac = LastFNOBlockWeightJacobian(fn, state, tiny.w0, fwd_batch=64)
    torch.manual_seed(0)
    est = _var_of_congruence_hutchinson(jac, cov, 4096)
    # Относительная ошибка на элемент ~ sqrt(2 / n_probe) ~ 2%.
    assert ((est - exact).abs() / exact).median() < 0.05
    assert abs(float(est.sum() / exact.sum()) - 1.0) < 0.02


def test_progress_values_match_final_spectrum(tiny):
    """Строка прогресса после последнего прохода печатает ровно итоговые S."""
    from luno_torch import ggn as ggn_mod

    G = dense_ggn(tiny.model_fn, tiny.xs(3), tiny.w0).detach()
    logged = []
    orig = ggn_mod._nystrom_values
    ggn_mod._nystrom_values = lambda Q, Y: logged.append(orig(Q, Y)) or logged[-1]
    try:
        U, S = randomized_eigh(_DenseOp(G), n=G.shape[0], rank=6, oversample=6,
                               power_iter=2)
    finally:
        ggn_mod._nystrom_values = orig
    assert len(logged) == 3
    assert close(logged[-1][:6], S, tol=1e-8)
    # Уже первый проход близок к истине по ведущему значению (в отличие от Ритца).
    top = torch.linalg.eigvalsh(G)[-1]
    assert abs(float(logged[0][0] / top) - 1.0) < 0.2
