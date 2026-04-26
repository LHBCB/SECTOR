# utility/scale_rule.py

def resolve_scale_profile(mode: str, n_obs: int, threshold: int) -> str:
    """
    Auto mode:
      - n_obs < 10000                    -> dense mode
      - 10000 <= n_obs < threshold       -> sparse mode with effective attr_graph_source='mlp'
      - n_obs >= threshold               -> sparse mode with effective attr_graph_source='raw'

    Manual overrides preserve existing semantics:
      - mode == 'off' -> dense mode
      - mode == 'on'  -> sparse mode (feature-graph source stays user-controlled)
    """
    mode = str(mode).strip().lower()
    n_obs = int(n_obs)
    threshold = int(threshold)

    if mode == "on":
        return "force_sparse"
    if mode == "off":
        return "dense_mode"
    if n_obs < 10000:
        return "dense_mode"
    if n_obs < threshold:
        return "sparse_mlp"
    return "sparse_raw"


def resolve_large_scale_mode(mode: str, n_obs: int, threshold: int) -> bool:
    return resolve_scale_profile(mode, n_obs, threshold) != "dense_mode"