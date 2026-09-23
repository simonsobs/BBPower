import numpy as np


class _Covariance:
    def __init__(self, covmat):
        self.covmat = covmat


class _Tracer:
    def __init__(self):
        self.nu = None
        self.bandpass = None
        self.ell = None
        self.beam = None
        self.quantity = 'cmb_polarization'


class _Window:
    def __init__(self, values, weight):
        self.values = values
        self.weight = weight


class _DataPoint(dict):
    pass


class NuMapSacc:
    """
    Minimal FITS-backed reader for NuMap SACC files.

    This implements only the subset of the sacc.Sacc API used by
    compsep_nopipe/compsep for external bandpower likelihoods.
    """
    def __init__(self, path):
        self.path = path
        self.tracers = {}
        self._windows = {}
        self._rows = []
        self._removed_tracers = set()
        self._removed_data_types = set()
        self._ell_min = None
        self._ell_max = None
        self._load_fits(path)
        self._refresh()

    @classmethod
    def load_fits(cls, path):
        return cls(path)

    def _load_fits(self, path):
        from astropy.io import fits

        with fits.open(path, memmap=True) as hdul:
            for hdu in hdul:
                sacctype = hdu.header.get('SACCTYPE')
                saccname = hdu.header.get('SACCNAME')
                if sacctype == 'tracer':
                    tracer = self.tracers.setdefault(saccname, _Tracer())
                    cols = hdu.data.columns.names
                    if 'nu' in cols:
                        tracer.nu = np.asarray(hdu.data['nu'], dtype=float)
                        tracer.bandpass = np.asarray(hdu.data['bandpass'],
                                                     dtype=float)
                    if 'ell' in cols:
                        tracer.ell = np.asarray(hdu.data['ell'], dtype=float)
                        tracer.beam = np.asarray(hdu.data['beam'], dtype=float)

                elif sacctype == 'window':
                    self._windows[str(saccname)] = _Window(
                        np.asarray(hdu.data['values'], dtype=float),
                        np.asarray(hdu.data['weight'], dtype=float),
                    )

                elif sacctype == 'data':
                    rows = hdu.data
                    data_type = str(saccname)
                    for i in range(len(rows)):
                        self._rows.append({
                            'data_type': data_type,
                            'tracer_0': str(rows['tracer_0'][i]),
                            'tracer_1': str(rows['tracer_1'][i]),
                            'value': float(rows['value'][i]),
                            'ell': float(rows['ell'][i]),
                            'window': str(rows['window'][i]),
                            'window_ind': int(rows['window_ind'][i]),
                            'sacc_ordering': int(rows['sacc_ordering'][i]),
                        })

                elif sacctype == 'covariance':
                    self._full_cov = np.asarray(hdu.data['row'], dtype=float)

        self._rows.sort(key=lambda row: row['sacc_ordering'])

    def _row_is_active(self, row):
        if row['data_type'] in self._removed_data_types:
            return False
        if row['tracer_0'] in self._removed_tracers:
            return False
        if row['tracer_1'] in self._removed_tracers:
            return False
        if self._ell_min is not None and row['ell'] < self._ell_min:
            return False
        if self._ell_max is not None and row['ell'] > self._ell_max:
            return False
        return True

    def _refresh(self):
        self._active_rows = [row for row in self._rows
                             if self._row_is_active(row)]
        self._active_orders = np.array([row['sacc_ordering']
                                        for row in self._active_rows],
                                       dtype=int)
        self._order_to_index = {
            row['sacc_ordering']: i for i, row in enumerate(self._active_rows)
        }
        self.mean = np.array([row['value'] for row in self._active_rows],
                             dtype=float)
        if hasattr(self, '_full_cov'):
            self.covariance = _Covariance(
                self._full_cov[np.ix_(self._active_orders,
                                      self._active_orders)]
            )
        self.data = []
        for row in self._active_rows:
            point = _DataPoint(row)
            point['window'] = self._windows[row['window']]
            self.data.append(point)

    def remove_tracers(self, tracers):
        self._removed_tracers.update(tracers)
        for tracer in tracers:
            self.tracers.pop(tracer, None)
        self._refresh()

    def remove_selection(self, data_type=None, ell__gt=None, ell__lt=None):
        if data_type is not None:
            self._removed_data_types.add(data_type)
        if ell__gt is not None:
            self._ell_max = ell__gt if self._ell_max is None else min(
                self._ell_max, ell__gt
            )
        if ell__lt is not None:
            self._ell_min = ell__lt if self._ell_min is None else max(
                self._ell_min, ell__lt
            )
        self._refresh()

    def get_tracer_combinations(self):
        pairs = []
        seen = set()
        for row in self._active_rows:
            pair = (row['tracer_0'], row['tracer_1'])
            if pair in seen:
                continue
            seen.add(pair)
            pairs.append(pair)
        return pairs

    def indices(self, data_type, tracers):
        tr1, tr2 = tracers
        inds = []
        for row in self._active_rows:
            if row['data_type'] != data_type:
                continue
            if row['tracer_0'] != tr1 or row['tracer_1'] != tr2:
                continue
            inds.append(self._order_to_index[row['sacc_ordering']])
        return np.array(inds, dtype=int)

    def get_ell_cl(self, data_type, tracer1, tracer2, return_cov=False):
        ind = self.indices(data_type, (tracer1, tracer2))
        rows = [self._active_rows[i] for i in ind]
        ell = np.array([row['ell'] for row in rows], dtype=float)
        cl = self.mean[ind]
        if return_cov:
            return ell, cl, self.covariance.covmat[np.ix_(ind, ind)]
        return ell, cl

    def get_bandpower_windows(self, indices):
        rows = [self._active_rows[i] for i in indices]
        if not rows:
            raise ValueError("No bandpower windows were requested")
        values = self._windows[rows[0]['window']].values
        weight = np.column_stack([
            self._windows[row['window']].weight[:, row['window_ind']]
            for row in rows
        ])
        return _Window(values, weight)
