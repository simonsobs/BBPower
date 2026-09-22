from fnmatch import fnmatchcase


def normalize_pol(pol):
    pol = str(pol).upper()
    if pol in ('T', '0'):
        return 'T'
    if pol in ('E', 'B'):
        return pol
    raise ValueError("Unknown polarization channel %s" % pol)


def normalize_spectrum_label(spec):
    spec = str(spec).lower().replace('cl_', '')
    spec = spec.replace('0', 't')
    if len(spec) != 2:
        raise ValueError("Spectrum labels must have two fields")
    return ''.join(normalize_pol(p) for p in spec)


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def normalize_spectrum_rules(config):
    rules = config.get('fit_spectrum_rules')
    if rules is None:
        rules = config.get('fit_spectra_rules')
    if rules is None:
        return []

    normalized = []
    for rule in rules:
        spectra = rule.get('spectra', rule.get('fit_spectra'))
        if spectra is None:
            raise ValueError("Each fit_spectrum_rules entry needs spectra")

        tracer1 = rule.get('tracer1', rule.get('tracers1'))
        tracer2 = rule.get('tracer2', rule.get('tracers2'))
        tracers = rule.get('tracers')
        if tracers is not None:
            if len(tracers) != 2:
                raise ValueError("fit_spectrum_rules tracers must have length 2")
            tracer1, tracer2 = tracers

        normalized.append({
            'name': rule.get('name'),
            'spectra': [normalize_spectrum_label(s) for s in _as_list(spectra)],
            'tracer1': [str(t).lower() for t in (_as_list(tracer1) or ['*'])],
            'tracer2': [str(t).lower() for t in (_as_list(tracer2) or ['*'])],
            'unordered': bool(rule.get('unordered', True)),
        })
    return normalized


def selected_spectra_from_config(config):
    rules = normalize_spectrum_rules(config)
    if rules:
        spectra = []
        for rule in rules:
            spectra.extend(rule['spectra'])
        return list(dict.fromkeys(spectra))

    fit_spectra = config.get('fit_spectra')
    if fit_spectra is not None:
        return list(dict.fromkeys(
            normalize_spectrum_label(s) for s in fit_spectra
        ))

    pols = [normalize_pol(p) for p in config['pol_channels']]
    return list(dict.fromkeys(p1 + p2 for p1 in pols for p2 in pols))


def use_cl_block(cl_name, config):
    spec = normalize_spectrum_label(cl_name)
    pols = [normalize_pol(p) for p in config['pol_channels']]
    if (spec[0] not in pols) or (spec[1] not in pols):
        return False
    return spec in selected_spectra_from_config(config)


def _matches_any(value, patterns):
    value = str(value).lower()
    return any(fnmatchcase(value, pattern) for pattern in patterns)


def rule_matches(rule, tracer1, tracer2, spectrum):
    spectrum = normalize_spectrum_label(spectrum)
    if spectrum not in rule['spectra']:
        return False

    forward = (_matches_any(tracer1, rule['tracer1']) and
               _matches_any(tracer2, rule['tracer2']))
    if forward:
        return True
    if not rule['unordered']:
        return False
    return (_matches_any(tracer1, rule['tracer2']) and
            _matches_any(tracer2, rule['tracer1']))
