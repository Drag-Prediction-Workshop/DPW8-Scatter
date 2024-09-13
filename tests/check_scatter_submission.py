#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 10:28:19 2024

@author: kevinholst
"""

import os
import sys
import re


def exception_handler(exception_type, exception, traceback):
    # All your trace are belong to us!
    print('********************************************************************************')
    print('')
    print(f'{exception_type.__name__}: {exception}')
    print('')
    print('********************************************************************************')


sys.excepthook = exception_handler


def parse_tecplot_file(filename):
    # open the file for reading
    with open(filename, 'r') as f:
        title = None
        variables = None
        aux = {}
        zones = {}
        zone = None

        # read line by line
        while True:
            line = f.readline()
            if not line:
                break   # EOF reached

            line = line.strip()

            if line == '' or line.startswith('#'):
                # we found a blank or comment line
                continue

            if line.upper().startswith('TITLE'):
                spl = line.find('=')
                title = line[spl+1:].strip().strip('"')
            elif line.upper().startswith('ZONE'):
                spl = line.find('=')
                zone_title = line[spl+1:].strip().strip('"')
                zones[zone_title] = dict(aux={}, data=[], info={})
                zone = zones[zone_title]
            elif line.upper().startswith('VARIABLES'):
                spl = line.find('=')
                names = line[spl+1:].strip()
                if names == '':
                    # variable names are on the following line
                    names = f.readline().strip()
                variables = re.findall('"(.*?)"', names)
            elif line.upper().startswith('DATASETAUXDATA'):
                key, value = line[14:].strip().split('=')
                key = key.strip().upper()
                value = value.strip()
                aux[key] = value
            elif line.upper().startswith('AUXDATA'):
                key, value = line[7:].strip().split('=')
                key = key.strip().upper()
                value = value.strip()
                zone['aux'][key] = value
            elif line[0].isdigit() or line[0] == '-':
                # must be a data line, store in zone_data under current zone_title.
                # assumes data items are separated by whitespace
                data = line.split()
                if len(data) != len(variables):
                    raise RuntimeError('Length of data does not match length of variables in '
                                       f'{filename}')
                zone['data'].append(data)
            else:
                # must be additional zone info needed for parsing
                for item in line.split(','):
                    key, value = item.strip().split('=')
                    key = key.strip().upper()
                    value = value.strip()
                    zone['info'][key] = value

    return dict(title=title, variables=variables, aux=aux, zones=zones)


# Checks the ONERA OAT15A submission file
def check_oat15a_file(filename):
    # filename contains the entire path, get the basename for checking
    basename = os.path.basename(filename)

    # Try grabbing the full id, with participant id and suffix, from the file name
    try:
        full_id = basename.split('_')[1][:-4]
    except:
        raise RuntimeError(f"Could not parse participant id from '{filename}' filename must be of "
                           "the form 'OAT15A_[participant id with suffix].dat")

    # Try splitting the full id into participant id and suffix
    try:
        pid, suffix = full_id.split('.')
    except:
        raise RuntimeError(f"Unable to parse participant id and suffix from '{filename}'")

    if not (pid.isnumeric() and suffix.isnumeric()):
        raise RuntimeError(f"Participant id and suffix must be numeric, found {pid}.{suffix}")

    # Read the tecplot file and check the data inside
    data = parse_tecplot_file(filename)
    if data['title'] != full_id:
        RuntimeError(f"Title inside '{filename}' does not match participant id and suffix")

    # More checks here...


# Checks file name and spawns additional checks based on the file type
def check_file(filename):
    if not filename.endswith('.dat'):
        raise RuntimeError(f"Only .dat files can be submitted, found '{filename}'")

    # filename contains the entire path, get the basename for checking
    basename = os.path.basename(filename)
    if basename.startswith('OAT15A'):
        check_oat15a_file(filename)
        return

    raise RuntimeError(f"Filename '{filename}' does not match any accepted filenames for Scatter "
                       "Working Group.")


if __name__ == '__main__':
    print(f'Modified files: {sys.argv[1:]}')
    for filename in sys.argv[1:]:
        print(f'checking {filename}')
        check_file(filename)
