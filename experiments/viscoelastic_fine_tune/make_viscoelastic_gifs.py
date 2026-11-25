"""Render physical fields from a validation-selected viscoelastic checkpoint."""
import argparse
from functools import lru_cache
import html
import json
from pathlib import Path
import time

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data._utils.collate import default_collate

from foundno_viscoelastic import build_model
from train_foundno_viscoelastic import evaluate, load_states, seed, write_json
from viscoelastic_data import FIELD_NAMES, ViscoelasticDataset, prepare_batch


@lru_cache(None)
def font(size):
    path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    return ImageFont.truetype(path, size) if Path(path).exists() else ImageFont.load_default()


def colors(values, low, high, cmap):
    scaled = np.clip((values - low) / max(high - low, 1e-12), 0, 1)
    return matplotlib.colormaps[cmap](scaled, bytes=True)[..., :3]


def palette():
    shades = [matplotlib.colormaps[c](np.linspace(0, 1, 80), bytes=True)[:, :3]
              for c in ('viridis', 'RdBu_r', 'inferno')]
    shades.append(np.repeat(np.linspace(0, 255, 16, dtype=np.uint8)[:, None], 3, axis=1))
    result = Image.new('P', (1, 1))
    result.putpalette(np.concatenate(shades).astype(np.uint8).ravel().tolist())
    return result


PALETTE = palette()


def row_spec(true, pred):
    lo, hi = float(min(true.min(), pred.min())), float(max(true.max(), pred.max()))
    cmap = 'viridis'
    if lo < 0 < hi:
        hi = max(abs(lo), abs(hi)); lo = -hi; cmap = 'RdBu_r'
    return lo, max(hi, lo + 1e-12), cmap


def error_percent(true, pred):
    floor = np.maximum(.01 * np.percentile(np.abs(true), 99, axis=(1, 2, 3)), 1e-7)
    error = 100 * np.abs(pred - true) / np.maximum(np.abs(true), floor[:, None, None, None])
    return error, floor


def draw_row(canvas, y, label, true, pred, error, spec, panel=176):
    draw = ImageDraw.Draw(canvas)
    lo, hi, cmap = spec
    left, gap = 176, 12
    draw.text((8, y + 8), label, fill='black', font=font(13))
    for column, (array, limits, cm) in enumerate(((true, (lo, hi), cmap),
                                                 (pred, (lo, hi), cmap),
                                                 (error, (0, 100), 'inferno'))):
        x = left + column * (panel + gap)
        # Flip only the display's vertical axis: x/y fields are not transposed.
        pixels = colors(array[::-1], *limits, cm)
        tile = Image.fromarray(pixels).resize((panel, panel), Image.Resampling.NEAREST)
        canvas.paste(tile, (x, y))
        gradient = np.tile(np.linspace(limits[0], limits[1], panel), (7, 1))
        canvas.paste(Image.fromarray(colors(gradient, *limits, cm)), (x, y + panel + 3))
        draw.text((x, y + panel + 11), f'{limits[0]:.3g}', fill='black', font=font(10))
        draw.text((x + panel - 48, y + panel + 11), f'{limits[1]:.3g}', fill='black', font=font(10))


def render(rows, title, destination, duration=450):
    """Each row has true/pred/error [T,Y,X], a fixed scale and an actual time vector."""
    panel, row_height = 176, 210
    width, top = 752, 104
    frames = []
    specs = [row_spec(r['true'], r['pred']) for r in rows]
    for t in range(20):
        canvas = Image.new('RGB', (width, top + row_height * len(rows) + 54), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 8), title, fill='black', font=font(14))
        draw.text((10, 52), f'Frame {t}/19 | relative physical time is shown in each row', fill='black', font=font(12))
        for col, text in enumerate(('True', 'Predict', 'Relative error (%)')):
            draw.text((176 + col * (panel + 12), 76), text, fill='black', font=font(13))
        for i, row in enumerate(rows):
            label = f"{row['label']}\ndt={row['times'][t]:.3g}"
            draw_row(canvas, top + i * row_height, label, row['true'][t], row['pred'][t],
                     row['error'][t], specs[i], panel)
        draw.text((10, canvas.height - 45), 'Error: 100*|pred-true|/max(|true|, eps); eps=max(1% P99(|true|), 1e-7).',
                  fill='black', font=font(11))
        draw.text((10, canvas.height - 25), 'Scales fixed over time. Error colors saturate at 100%; metrics are not clipped.',
                  fill='black', font=font(11))
        if t == 10:
            canvas.save(destination.with_suffix('.png'))
        frames.append(canvas.quantize(palette=PALETTE, dither=Image.Dither.NONE))
    frames[0].save(destination, save_all=True, append_images=frames[1:], loop=0,
                   duration=duration, optimize=False, disposal=2)
    with Image.open(destination) as check:
        assert check.n_frames == 20 and check.info['loop'] == 0


def gallery(output, manifest):
    cards = []
    for split in ('train', 'valid', 'test'):
        cards.append(f'<h2>{split}</h2><p><a href="{split}/overview_velocity_y.gif">Animated overview: velocity_y, seven regimes</a></p>')
        cards.append('<div class="grid">')
        for item in manifest['samples']:
            if item['split'] != split:
                continue
            gif = html.escape(item['gif'])
            preview = html.escape(item['preview'])
            cards.append(f'<article><h3>{html.escape(item["regime"])}</h3><p>Trajectory {item["trajectory"]}, VRMSE={item["vrmse_future19"]:.4f}</p><a href="{gif}"><img loading="lazy" src="{preview}" alt="True, predict, error; click to animate"></a><p><a href="{gif}">Open 20-frame GIF</a></p></article>')
        cards.append('</div>')
    body = ''.join(cards)
    (output / 'index.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>FoundNO viscoelastic GIFs</title>
<style>body{{font:16px system-ui;max-width:1300px;margin:30px auto;padding:0 16px;background:#f6f7fb;color:#182233}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px}}article{{background:white;padding:16px;border-radius:10px}}img{{width:100%;height:auto}}a{{color:#1754a3}}</style>
<h1>FoundNO: viscoelastic_instability_v2</h1><p>Best validation VRMSE checkpoint, epoch {manifest['checkpoint_epoch']}. One randomly selected trajectory per regime and split, seed {manifest['seed']}. Click an image to open the GIF.</p>
<p>Columns: True / Predict / relative error (%). All 8 fields in physical units, frames 0..19, resolution 128x128. True/Pred share a scale fixed over all frames. Error = 100*abs(pred-true)/max(abs(true),epsilon), epsilon=max(1% P99(abs(true)),1e-7) per field and trajectory. Error colors saturate at 100%; this percentage differs from VRMSE.</p>{body}''', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_num_threads(4)
    seed(args.seed)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
    config, stats, epoch = checkpoint['config'], checkpoint['normalization'], checkpoint['epoch']
    model, _ = build_model(config)
    load_states(model, checkpoint)
    model.to('cuda').eval()
    manifest = {'checkpoint': str(checkpoint_path), 'checkpoint_epoch': epoch,
                'checkpoint_selection': 'minimum validation VRMSE; no selection by test error',
                'seed': args.seed, 'sample_selection': 'one random trajectory per split and regime; no error filtering',
                'field_order': FIELD_NAMES, 'time_indices': list(range(20)),
                'percent_error_formula': '100*abs(pred-true)/max(abs(true),epsilon)',
                'epsilon': 'max(1% P99(abs(true)) over T,X,Y per field and trajectory, 1e-7)',
                'error_color_scale_percent': [0, 100], 'samples': []}
    rng = np.random.default_rng(args.seed)
    datasets = {}
    print('MODEL_READY epoch=' + str(epoch), flush=True)
    for split in ('train', 'valid', 'test'):
        records = json.loads((checkpoint_path.parent / 'splits' / f'{split}.json').read_text())
        ds = ViscoelasticDataset(records, config['data']['resolution'], Path(config['data']['cache_dir']) / split)
        datasets[split] = ds
        folder = output / split; folder.mkdir(exist_ok=True)
        overview = []
        expected = None
        if split == 'valid':
            expected = np.load(checkpoint_path.parent / 'validation' / f'epoch_{epoch:04d}' / 'vrmse.npz')['vrmse']
        for regime in sorted({r['regime'] for r in records}):
            index = int(rng.choice([i for i,r in enumerate(records) if r['regime'] == regime]))
            record = records[index]
            with torch.inference_mode():
                x, target, true, mean, std = prepare_batch(default_collate([ds[index]]), stats, 'cuda')
                pred = model(x) * std + mean
                score = ((pred-true).square().mean(dim=(-2,-1))/(true.var(dim=(-2,-1),unbiased=True)+1e-7)).sqrt()[0]
                assert torch.isfinite(pred).all()
                score = score.cpu().numpy()
                if expected is not None:
                    np.testing.assert_allclose(score, expected[index], rtol=2e-3, atol=3e-4)
                true, pred = true[0].cpu().numpy(), pred[0].cpu().numpy()
            error, floors = error_percent(true, pred)
            destination = folder / f'{regime}_trajectory_{record["trajectory"]:02d}_epoch{epoch}.gif'
            info = {'split': split, 'regime': regime, 'dataset_index': index,
                    'trajectory': record['trajectory'], 'source_file': record['file'],
                    'relative_times': record['relative_times'], 'vrmse_future19': float(score[:,1:].mean()),
                    'vrmse_by_field': dict(zip(FIELD_NAMES, map(float, score[:,1:].mean(axis=1)))),
                    'epsilon_by_field': dict(zip(FIELD_NAMES, map(float, floors))),
                    'gif': destination.relative_to(output).as_posix(),
                    'preview': destination.with_suffix('.png').relative_to(output).as_posix()}
            np.savez_compressed(destination.with_suffix('.npz'), truth=true, prediction=pred,
                                time_indices=np.arange(20), relative_times=record['relative_times'],
                                field_names=np.array(FIELD_NAMES))
            rows = [{'label': name, 'true': true[c], 'pred': pred[c], 'error': error[c],
                     'times': record['relative_times']} for c,name in enumerate(FIELD_NAMES)]
            render(rows, f'FoundNO | epoch {epoch} | {split} | {regime}\nTrajectory {record["trajectory"]} | future VRMSE={info["vrmse_future19"]:.4f}', destination)
            overview.append(dict(rows[3], label=f'{regime}\nVRMSE {info["vrmse_future19"]:.3f}'))
            manifest['samples'].append(info)
            write_json(output/'manifest.json', manifest)
            print('GIF ' + json.dumps(info), flush=True)
        render(overview, f'FoundNO | epoch {epoch} | {split} | velocity_y: all seven regimes',
               folder / 'overview_velocity_y.gif')
    # Training has stopped; evaluate the validation-selected checkpoint on all 22 held-out trajectories.
    test_output = checkpoint_path.parent / 'final_test' / 'best_vrmse'
    summary = evaluate(model, datasets['test'], stats, 'cuda', test_output)
    summary['checkpoint_epoch'] = epoch
    write_json(test_output/'summary.json', summary)
    manifest['full_test'] = summary
    write_json(output/'manifest.json', manifest)
    gallery(output, manifest)
    write_json(output/'completed.json', {'gif_count': len(manifest['samples'])+3,
               'checkpoint_epoch': epoch, 'seconds': time.perf_counter()-started,
               'all_gifs_checked': '20 frames, infinite loop; validation predictions match stored field/time metrics',
               'test_vrmse_future19': summary['vrmse_future19']})
    print('COMPLETED ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
