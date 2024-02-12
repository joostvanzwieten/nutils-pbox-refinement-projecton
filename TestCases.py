


def CaseLoader(settings: dict, CaseName):


    if CaseName == "Case1":
        settings['Re'] = 2e3
        settings['Rm'] = 1e2
        settings['S'] = 0.0125*settings['Rm']

        Conditions = {'uwall'   : "np.stack([topo.boundary.indicator('top'), 0])",
                      'Bwall'   : "np.stack([1, 0])",
                      'f'       : "np.stack([0, 0])",
                      'g'       : "np.stack([0, 0])"
                      }

    elif CaseName == "Case2":
        settings['Re'] = 2e3
        settings['Rm'] = 1e2
        settings['S'] = 0.0245*settings['Rm']

        Conditions = {'uwall'   : "np.stack([topo.boundary.indicator('top'), 0])",
                      'Bwall'   : "np.stack([1, 0])",
                      'f'       : "np.stack([0, 0])",
                      'g'       : "np.stack([0, 0])"
                      }

    elif CaseName == "Case3":
        settings['Re'] = 2e3
        settings['Rm'] = 1e2
        settings['S'] = 0.05*settings['Rm']

        Conditions = {'uwall'   : "np.stack([topo.boundary.indicator('top'), 0])",
                      'Bwall'   : "np.stack([1, 0])",
                      'f'       : "np.stack([0, 0])",
                      'g'       : "np.stack([0, 0])"
                      }
    elif CaseName == "HighRey":
        settings['Re'] = 1e3
        settings['Rm'] = 1e3
        settings['S'] = 1*settings['Rm']

        Conditions = {'uwall'   : "np.stack([topo.boundary.indicator('top'), 0])",
                      'Bwall'   : "np.stack([1, 0])",
                      'f'       : "np.stack([0, 0])",
                      'g'       : "np.stack([0, 0])"
                      }

    elif CaseName == "LowRey":
        settings['Re'] = 1e1
        settings['Rm'] = 1e1
        settings['S'] = 1 * settings['Rm']

        Conditions = {'uwall': "np.stack([topo.boundary.indicator('top'), 0])",
                      'Bwall': "np.stack([1, 0])",
                      'f': "np.stack([0, 0])",
                      'g': "np.stack([0, 0])"
                      }

    elif CaseName == "LidDrivenNoMHD":
        settings['Re'] = 1e3
        settings['Rm'] = 1
        settings['S'] = 0*settings['Rm']

        Conditions = {'uwall'   : "np.stack([topo.boundary.indicator('top'), 0])",
                      'Bwall'   : "np.stack([0, 0])",
                      'f'       : "np.stack([0, 0])",
                      'g'       : "np.stack([0, 0])"
                      }

    else:
        settings['Re'] = 1e0
        settings['Rm'] = 1e0
        settings['S'] = 1 * settings['Rm']

        Conditions = {'uwall'   : "np.stack([topo.boundary.indicator('top'), 0])",
                      'Bwall'   : "np.stack([1, 0])",
                      'f'       : "np.stack([0, 0])",
                      'g'       : "np.stack([0, 0])"
                      }

    settings['Conditions'] = Conditions


    return settings