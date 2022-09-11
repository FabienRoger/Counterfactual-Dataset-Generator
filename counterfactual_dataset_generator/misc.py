def overwrite_fire_help_text():  # type: ignore
    import inspect
    import fire  # type: ignore
    from fire import inspectutils  # type: ignore
    from fire.core import _ParseKeywordArgs  # type: ignore

    def NewHelpText(component, trace=None, verbose=False):
        if callable(component):
            return component.__doc__
        elif isinstance(component, dict):
            docs = {k: v.__doc__.split("\n")[0] for k, v in component.items()}
            return "COMMANDS\n" + "\n".join(f"  {k}\n   {doc}" for k, doc in docs.items())
        else:
            return ""

    fire.helptext.HelpText = NewHelpText

    def _NewIsHelpShortcut(component_trace, remaining_args):
        show_help = False
        if remaining_args:
            target = remaining_args[0]
            if target in ("-h", "--help"):
                # Check if --help would be consumed as a keyword argument, or is a member.
                component = component_trace.GetResult()
                if inspect.isclass(component) or inspect.isroutine(component):
                    fn_spec = inspectutils.GetFullArgSpec(component)
                    _, remaining_kwargs, _ = _ParseKeywordArgs(remaining_args, fn_spec)
                    show_help = target in remaining_kwargs
                else:
                    members = dict(inspect.getmembers(component))
                    show_help = target not in members
        if show_help:
            component_trace.show_help = True
        return show_help

    fire.core._IsHelpShortcut = _NewIsHelpShortcut
