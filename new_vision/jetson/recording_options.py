"""Names shared by the button config, vision recorder and policy launcher."""

LOG_ITEMS = ('vision', 'control', 'video', 'motor', 'policy', 'shape', 'loss')


def parse_log_items(value):
    names = str(value).lower().replace(',', ' ').split()
    if names == ['all']:
        return LOG_ITEMS
    if not names or names == ['none']:
        return ()
    unknown = set(names) - set(LOG_ITEMS)
    if unknown:
        raise ValueError('LOG has invalid items: ' + ', '.join(sorted(unknown)) +
                         '; use ' + ' '.join(LOG_ITEMS) + ', all, or none')
    return tuple(name for name in LOG_ITEMS if name in names)
