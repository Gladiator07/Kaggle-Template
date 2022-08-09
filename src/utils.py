import slack
from types import SimpleNamespace


class SlackNotifier:
    """
    Notifies about important events to a dedicated slack channel
    """

    def __init__(self, slack_token: str, channel: str):
        self.slack_token = slack_token
        self.channel = channel
        self.client = slack.WebClient(token=self.slack_token)

    def notify(self, message: str):
        self.client.chat_postMessage(channel=self.channel, text=message)


def overwrite_config_from_cmd_args(args: list, cfg: SimpleNamespace) -> SimpleNamespace:
    """
    Overwrites configuration in `.py` configuration files from command-line
    Args:
        args (list): other arguments passed to command-line returned by function `parser.parse_known_args`
        cfg (SimpleNamespace): config object
    Returns:
        cfg (SimpleNamespace): modified config object
    """
    assert (
        len(args) > 1
    ), "Args with length < 1 can't be used for overwriting config object"
    print("#" * 50)
    print(f"\nOVERWRITING CONFIG FROM CMD ARGS: ")
    args = {k.replace("-", ""): v for k, v in zip(args[1::2], args[2::2])}
    for key in args:
        if key in cfg.__dict__:
            print(f"cfg.{key}: {cfg.__dict__[key]} -> {args[key]}")
            cfg_type = type(cfg.__dict__[key])
            if cfg_type == bool:
                cfg.__dict__[key] = args[key] == "True"
            elif cfg_type == type(None):
                cfg.__dict__[key] = args[key]
            else:
                cfg.__dict__[key] = cfg_type(args[key])
        elif key.split(".")[0] in cfg.__dict__:
            # multi-level keys
            assert "." in key
            # trainer.max_epochs -> ['trainer', 'max_epochs']
            prim_key, sub_key = key.split(".")
            if sub_key not in cfg.__dict__[prim_key].__dict__:
                raise KeyError(
                    f"Unknown subkey `{sub_key}` for key `{prim_key}` passed"
                )
            print(
                f"{prim_key}.{sub_key}: {cfg.__dict__[prim_key].__dict__[sub_key]} -> {args[key]}"
            )
            cfg_type = type(cfg.__dict__[prim_key].__dict__[sub_key])
            if cfg_type == bool:
                cfg.__dict__[prim_key].__dict__[sub_key] = args[key] == "True"
            elif cfg_type == type(None):
                cfg.__dict__[prim_key].__dict__[sub_key] = args[key]
            else:
                cfg.__dict__[prim_key].__dict__[sub_key] = cfg_type(args[key])
        else:
            raise KeyError(f"Unknown key `{key}` passed")

    print("\n" + "#" * 50)

    return cfg
