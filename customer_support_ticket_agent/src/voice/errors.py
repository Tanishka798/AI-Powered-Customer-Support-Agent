class VoiceInputError(ValueError):
    """The supplied audio or text can't be used (empty, unsupported, or unreadable).

    Subclasses ValueError so ``VoicePipeline``'s own input checks and adapter
    input problems are handled the same way: as a client error, not a provider failure.
    """
