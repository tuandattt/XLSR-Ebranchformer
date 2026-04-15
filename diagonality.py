from collections import defaultdict
import argparse
import torch
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from data_utils import Dataset_eval
from model import Model
from utils import read_metadata


def diagonality(att):
    if isinstance(att, torch.Tensor):
        att = att.detach().cpu().numpy()

    bs, nheads, len_out, len_in = att.shape

    rel_distance = np.zeros((len_out, len_in), dtype=np.float32)
    for i in range(len_out):
        for j in range(len_in):
            rel_distance[i, j] = np.abs(i - j)

    denom = rel_distance.max(-1)
    denom[denom == 0] = 1.0

    result = (1.0 - (att * rel_distance).sum(-1) / denom).mean(-1)
    return result


def format_array(arr):
    return np.array2string(
        arr,
        formatter={"float_kind": lambda x: f"{x:.6f}"},
        separator=","
    )


def run_diagonality_eval(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=10, shuffle=False, drop_last=False)
    model.eval()
    diag_dict = defaultdict(list)

    with torch.no_grad():
        for batch_x, utt_id in tqdm(data_loader):
            batch_x = batch_x.to(device)
            _, attn_score = model(batch_x)

            for layer_idx, att in enumerate(attn_score):
                if att is None:
                    continue
                diag = diagonality(att)
                diag_dict[f"layer_{layer_idx}"].append(diag)

    with open(save_path, "w", encoding="utf-8") as fh:
        for layer_name, vals in diag_dict.items():
            vals = np.concatenate(vals, axis=0)
            fh.write(
                f"{layer_name}: nsamples={vals.shape[0]}, mean={format_array(vals.mean(0))}, std={format_array(vals.std(0))}\n"
            )


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Diagonality evaluation')
#     parser.add_argument('--database_path', type=str, default='ASVspoof_database/')
#     parser.add_argument('--protocols_path', type=str, default='ASVspoof_database/')
#     parser.add_argument('--emb-size', type=int, default=144)
#     parser.add_argument('--heads', type=int, default=4)
#     parser.add_argument('--kernel_size', type=int, default=31)
#     parser.add_argument('--num_encoders', type=int, default=4)
#     parser.add_argument('--ckpt_path', type=str, default=None)
#     parser.add_argument('--save_dir', type=str, default='Diagonality')
#     parser.add_argument('--tracks', type=str, default='LA,DF')

#     args = parser.parse_args()
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     model = Model(args, device)
#     # model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
#     model = model.to(device)
#     model.eval()

#     os.makedirs(args.save_dir, exist_ok=True)

#     eval_tracks = [x.strip() for x in args.tracks.split(',') if x.strip()]

#     # for tracks in eval_tracks:
#     #     prefix = f'ASVspoof_{tracks}'
#     #     prefix_2021 = f'ASVspoof2021.{tracks}'

#     #     meta_path = os.path.join(
#     #         args.protocols_path,
#     #         # f'{tracks}/{prefix}_cm_protocols/{prefix_2021}.cm.eval.trl.txt'
#     #         f'{tracks}/{prefix}_cm_protocols/{prefix_2021}.cm.dev.trl.txt'

#     #     )

#     #     base_dir = os.path.join(
#     #         args.database_path,
#     #         # f'{tracks}/ASVspoof2021_{tracks}_eval/'
#     #         f'{tracks}/ASVspoof2021_{tracks}_dev/'
#     #     )
#     for tracks in eval_tracks:
#         prefix_2019 = f'ASVspoof2019_{tracks}'
#         prefix_file = f'ASVspoof2019.{tracks}'

#         meta_path = os.path.join(
#             args.protocols_path,
#             f'{tracks}/{tracks}/{prefix_2019}_cm_protocols/{prefix_file}.cm.dev.trl.txt'
#         )

#         base_dir = os.path.join(
#             args.database_path,
#             f'{tracks}/{tracks}/ASVspoof2019_{tracks}_dev/'
#         )
#         _, file_eval = read_metadata(dir_meta=meta_path, is_eval=False)
#         eval_set = Dataset_eval(
#             list_IDs=file_eval,
#             base_dir=base_dir,
#             track=tracks
#         )

#         ckpt_name = args.ckpt_path.replace('/', '_') if args.ckpt_path else "no_ckpt"
#         save_path = os.path.join(args.save_dir, f'{tracks}_{ckpt_name}.txt')

#         run_diagonality_eval(eval_set, model, device, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diagonality evaluation')

    parser.add_argument(
        '--database_path',
        type=str,
        default='/home4/datpt/data_spoof/ASVspoof2021'
    )
    parser.add_argument(
        '--protocols_path',
        type=str,
        default='/home4/datpt/data_spoof/ASVspoof2021'
    )
    parser.add_argument(
        '--emb-size',
        dest='emb_size',
        type=int,
        default=144
    )
    parser.add_argument(
        '--heads',
        type=int,
        default=4
    )
    parser.add_argument(
        '--kernel_size',
        type=int,
        default=31
    )
    parser.add_argument(
        '--num_encoders',
        type=int,
        default=4
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='/home4/datpt/thuhb/XLSR-Ebranchformer/diagonality/2021_DF_Ebranch'
    )
    parser.add_argument(
        '--tracks',
        type=str,
        default='DF'
    )
    parser.add_argument(
        '--ckpt_path',
        type=str,
        # Conformer checkpoint DF
        # default='/home4/datpt/thuhb/checkpoints/avg_5_best.pth'
          # Conformer checkpoint LA
        # default='/home4/datpt/thuhb/checkpoints/best_4.pth'
        # Ebranchformer checkpoint
        default=None
        
    )
    

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Using device:', device)
    print('database_path :', args.database_path)
    print('protocols_path:', args.protocols_path)
    print('save_dir      :', args.save_dir)
    print('tracks        :', args.tracks)

    model = Model(args, device)

    if args.ckpt_path is not None:
        model.load_state_dict(torch.load(args.ckpt_path, map_location=device))

    model = model.to(device)
    model.eval()

    os.makedirs(args.save_dir, exist_ok=True)

    eval_tracks = [x.strip() for x in args.tracks.split(',') if x.strip()]

    for tracks in eval_tracks:
        prefix_2021 = f'ASVspoof2021_{tracks}'
        prefix_file = f'ASVspoof2021.{tracks}'

        meta_path = os.path.join(
            args.protocols_path,
            f'{prefix_2021}_eval/{prefix_file}.cm.eval.trl.txt'
        )

        base_dir = os.path.join(
            args.database_path,
            f'{prefix_2021}_eval/flac'
        )

        print(f'\n[Track {tracks}]')
        print('meta_path:', meta_path)
        print('base_dir :', base_dir)

        file_eval = read_metadata(dir_meta=meta_path, is_eval=True)

        eval_set = Dataset_eval(
            list_IDs=file_eval,
            base_dir=base_dir,
            track=tracks
        )

        ckpt_name = args.ckpt_path.replace('/', '_') if args.ckpt_path else "no_ckpt"
        save_path = os.path.join(args.save_dir, f'{tracks}_{ckpt_name}.txt')

        run_diagonality_eval(eval_set, model, device, save_path)

        print('Saved to:', save_path)