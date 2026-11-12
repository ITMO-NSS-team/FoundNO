from __future__ import annotations

import torch

from luno_torch.kfac import KFACLastBlock

from .conftest import close, dense_ggn, dense_jacobian


def _layout(tiny):
    """Размеры блоков плоского w = [Re R, Im R, W, b] и число удержанных мод."""
    R, W, b = tiny.wrapper.params
    n_in, n_out = R.shape[0], R.shape[1]
    K = R[0, 0].numel()
    return n_in, n_out, K, R.numel(), W.numel()


def _mode_index(n_in, n_out, K, k):
    """Плоские индексы R[i, o, k] для всех (i, o), в порядке i * n_out + o."""
    io = torch.arange(n_in * n_out)
    return io * K + k


def test_weight_gradient_is_outer_product(tiny):
    """grad_R = g_hat (x) conj(z_hat): на этом стоит вся факторизация."""
    x = tiny.xs(1)[0]
    ctx = KFACLastBlock(tiny.affine, tiny.model_fn, tiny.w0)
    n_in, n_out, K, n_r, _ = _layout(tiny)
    u = torch.randn(tiny.model_fn(x, tiny.w0).shape, dtype=tiny.w0.dtype)

    w = tiny.w0.clone().requires_grad_(True)
    (grad_w,) = torch.autograd.grad((tiny.model_fn(x, w) * u).sum(), w)
    grad_R = torch.complex(grad_w[:n_r], grad_w[n_r: 2 * n_r]).reshape(n_in, n_out, K)

    # g_hat и z_hat тем же приёмом, что в accumulate().
    from luno_torch.affine import BoundAffine

    bound = BoundAffine(tiny.affine, x, tiny.model_fn, tiny.w0)
    rec = {}

    def contract(x_hat, weight, orig):
        out = orig(x_hat, weight, separable=False)
        rec["zhat"] = x_hat.detach()
        rec["d"] = torch.zeros_like(out, requires_grad=True)
        return out + rec["d"]

    p = tiny.affine
    s = ctx._conv_with(bound.z, contract) + p.conv.transform(p.skip(bound.z), output_shape=None)
    out = ctx._tail(s, bound.cskip)
    (g_hat,) = torch.autograd.grad((out * u).sum(), rec["d"])
    outer = torch.einsum("ok,ik->iok", g_hat[0].flatten(1), rec["zhat"][0].flatten(1).conj())
    assert close(outer, grad_R)


def test_kronecker_is_exact_for_one_sample(tiny):
    """N = 1, точный B: циркулярная часть плотного GGN моды == A_k (x) B_k / 2."""
    xs = tiny.xs(1)
    n_in, n_out, K, n_r, _ = _layout(tiny)
    G = dense_ggn(tiny.model_fn, xs, tiny.w0).detach()
    ctx = KFACLastBlock(tiny.affine, tiny.model_fn, tiny.w0)
    acc = ctx.new_accumulator()
    ctx.accumulate(xs[0], acc, exact=True)

    for k in (0, 1, K // 2, K - 1):
        re = _mode_index(n_in, n_out, K, k)
        im = re + n_r
        M11, M12 = G[re][:, re], G[re][:, im]
        M21, M22 = G[im][:, re], G[im][:, im]
        gamma_half = 0.5 * torch.complex(M11 + M22, M21 - M12)
        kron = 0.5 * torch.kron(acc["A"][k], acc["B"][k])
        assert close(kron, gamma_half.to(kron.dtype), tol=1e-4)

    # bias: фактор точен при любом N.
    _, _, _, _, n_w = _layout(tiny)
    b_idx = torch.arange(2 * n_r + n_w, tiny.w0.numel())
    assert close(acc["Bb"], G[b_idx][:, b_idx], tol=1e-4)


def _dense_kfac_precision(tiny, terms, prior_prec):
    """Плотная прецизионная матрица KFAC-апостериора в координатах плоского w."""
    n_in, n_out, K, n_r, n_w = _layout(tiny)
    d = tiny.w0.numel()
    f64 = torch.float64
    H = prior_prec * torch.eye(d, dtype=f64)
    N, X = terms.n_samples, terms.n_points
    A = (terms.QA * terms.a.unsqueeze(-2).to(terms.QA.dtype)) @ terms.QA.mH
    B = (terms.QB * terms.b.unsqueeze(-2).to(terms.QB.dtype)) @ terms.QB.mH
    for k in range(K):
        Hk = torch.kron(A[k], B[k]).to(torch.complex128) / (2.0 * N)
        re = _mode_index(n_in, n_out, K, k)
        im = re + n_r
        H[re[:, None], re] += Hk.real
        H[im[:, None], im] += Hk.real
        H[im[:, None], re] += Hk.imag
        H[re[:, None], im] -= Hk.imag
    AW = (terms.QAW * terms.aW) @ terms.QAW.T
    BW = (terms.QBW * terms.bW) @ terms.QBW.T
    w_idx = torch.arange(2 * n_r, 2 * n_r + n_w)  # W[o, i, 1] -> o * n_in + i
    H[w_idx[:, None], w_idx] += torch.kron(BW, AW).to(f64) / float(N * X)
    b_idx = torch.arange(2 * n_r + n_w, d)
    Bb = (terms.Qb * terms.bb) @ terms.Qb.T
    H[b_idx[:, None], b_idx] += Bb.to(f64)
    return H


def test_function_space_sampler_matches_dense_posterior(tiny):
    """Сэмплер в пространстве суммы блока даёт diag(J H^{-1} J^T) плотного KFAC."""
    xs = tiny.xs(3)
    ctx = KFACLastBlock(tiny.affine, tiny.model_fn, tiny.w0)
    terms = ctx.fit(xs, n_mc=6, seed=0)
    x = tiny.xs(1, seed=7)[0]
    J = dense_jacobian(tiny.model_fn, x, tiny.w0).detach()
    precs = [0.05, 3.0]
    gen = torch.Generator().manual_seed(0)
    mean, var = ctx.mean_and_var(x, terms, precs, n_probe=6000, chunk=500, generator=gen)
    assert close(mean, tiny.model_fn(x, tiny.w0).reshape(-1))
    for i, lam in enumerate(precs):
        H = _dense_kfac_precision(tiny, terms, lam)
        exact = ((J @ torch.linalg.inv(H)) * J).sum(dim=1)
        rel = (var[i] - exact).abs() / exact
        # Ошибка Монте-Карло на элемент ~ sqrt(2 / n_probe) ~ 1.8%.
        assert rel.median() < 0.03 and rel.max() < 0.12
        assert abs(float(var[i].sum() / exact.sum()) - 1.0) < 0.01
    # Больше prior_prec -- уже апостериор.
    assert bool((var[1] < var[0]).all())


def test_kfac_terms_roundtrip_and_shapes(tiny):
    import pickle

    ctx = KFACLastBlock(tiny.affine, tiny.model_fn, tiny.w0)
    terms = ctx.fit(tiny.xs(2), n_mc=2, seed=1)
    n_in, n_out, K, _, _ = _layout(tiny)
    assert terms.QA.shape == (K, n_in, n_in) and terms.QB.shape == (K, n_out, n_out)
    assert terms.n_samples == 2 and bool((terms.a >= 0).all()) and bool((terms.b >= 0).all())
    back = pickle.loads(pickle.dumps(terms.to(device="cpu")))
    assert close(back.QB, terms.QB) and back.mode_shape == terms.mode_shape
