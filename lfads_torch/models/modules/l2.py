import torch


def compute_recurrent_size(hps):
    recurrent_units_and_weights = [
        (hps.ic_enc_dim, hps.l2_ic_enc_scale),
        (hps.ic_enc_dim, hps.l2_ic_enc_scale),
        (hps.ci_enc_dim, hps.l2_ci_enc_scale),
        (hps.ci_enc_dim, hps.l2_ci_enc_scale),
        (hps.gen_dim, hps.l2_gen_scale),
        (hps.con_dim, hps.l2_con_scale),
    ]
    model_recurrent_size = 0
    for units, weight in recurrent_units_and_weights:
        if weight > 0:
            model_recurrent_size += 3 * units ** 2
    return model_recurrent_size


def compute_l2_penalty(lfads, hps):
    recurrent_kernels_and_weights = [
        (lfads.encoder.ic_enc.fwd_gru.weight_hh_l0, hps.l2_ic_enc_scale),
        (lfads.encoder.ic_enc.bwd_gru.weight_hh_l0, hps.l2_ic_enc_scale),
        (lfads.decoder.rnn.gen_cell.weight_hh_l0, hps.l2_gen_scale),
    ]
    if lfads.use_con:
        recurrent_kernels_and_weights.append(
            [
                (lfads.encoder.ci_enc.fwd_gru.weight_hh_l0, hps.l2_ci_enc_scale),
                (lfads.encoder.ci_enc.bwd_gru.weight_hh_l0, hps.l2_ci_enc_scale),
                (lfads.decoder.rnn.con_cell.weight_hh_l0, hps.l2_con_scale),
            ]
        )
    l2_penalty = 0.0
    for kernel, weight in recurrent_kernels_and_weights:
        # TODO: This calculation may not match LFADS exactly
        l2_penalty += weight * torch.norm(kernel, 2) ** 2
    recurrent_size = compute_recurrent_size(hps)
    return l2_penalty / (recurrent_size + 1e-8)
