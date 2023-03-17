def linkcode_resolve(domain, info):
    if domain != 'py':
        return None

    if not info['module']:
        return None

    filename = info['module'].replace('.', '/')
    return f'https://github.com/medusa-4D/medusa/blob/master/{filename}.py'
