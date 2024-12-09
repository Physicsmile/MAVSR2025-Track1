
import torch


def init_model(cfg):
    # init frontend
    if cfg.frontend_type == "Conv3dResNet":
        from models import frontends
        frontend_args = cfg.frontend_args or {}
        frontend = frontends.Conv3dResNet(**frontend_args)
    else:
        raise NotImplementedError()

    # init encoder
    if cfg.encoder_type == "Transformer":
        if cfg.encoder_args.name == "UniEncoder":
            from models.transformer import UniEncoder as Encoder
        elif cfg.encoder_args.name == "TMEncoder":
            from models.transformer import TMEncoder as Encoder
        else:
            raise NotImplementedError
        tm_args = cfg.encoder_args.args or {}
        encoder = Encoder(**tm_args)
    elif cfg.encoder_type == "Conformer":
        if cfg.encoder_args.name == "ConformerEncoder":
            from models.conformer import ConformerEncoder as Encoder
        else:
            raise NotImplementedError()
        conformer_args = cfg.encoder_args.args or {}
        encoder = Encoder(**conformer_args)
    else:
        raise NotImplementedError()

    # (optional) init decoder
    if cfg.decoder_type is None: 
        decoder = None
    else:
        if cfg.decoder_type == "Transformer":
            if cfg.decoder_args.name == "UniDecoder":
                from models.transformer import UniDecoder as Decoder
            elif cfg.decoder_args.name == "TMDecoder":
                from models.transformer import TMDecoder as Decoder
            else:
                raise NotImplementedError()
            tm_args = cfg.decoder_args.args or {}
            decoder = Decoder(**tm_args)
        else:
            raise NotImplementedError()

    m_args = cfg.model_args
    if cfg.model_type == "TMDecoding":
        from .tm_decoding_model import TMDecoding
        model = TMDecoding(
            frontend, encoder, decoder,
            m_args.frontend_dim, m_args.enc_in_dim, m_args.enc_out_dim,
            m_args.dec_in_dim, m_args.dec_out_dim, m_args.vocab
            )    
    else:
        raise NotImplementedError()
        
    return model

def init_beam_search_decoder(cfg, model, tokenizer):
    from models.beam_search import BeamSearch

    return BeamSearch(
        model=model, tokenizer=tokenizer,
        max_decode_len=cfg.max_decode_len, bms=cfg.beam_size,
        sos=cfg.sos, eos=cfg.eos, blank=cfg.blank,
        ctc_beta=cfg.ctc_beta, lb_beta=cfg.lb_beta,
    )
