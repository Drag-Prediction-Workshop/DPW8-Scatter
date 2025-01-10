#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 10:28:19 2024

@author: kevinholst
"""

import os
import sys
import re
from pathlib import PurePath


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
                value = value.strip().strip('"')
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


def get_id_from_path(path):
    # get the participant ID from the directory structure
    try:
        pid = path.parts[-3].split('_')[0]
        if not pid.isnumeric():
            raise RuntimeError(f"Participant id must be numeric, found {pid}")
    except:
        raise RuntimeError(f"Could not parse participant id from '{path}'. File must be in "
                            "directory structure of the form 'TestCase1a/[participant_id]_"
                            "[participant_info]/[submission_id]_[submission_info]/[filename].dat")

    # get the submission ID from the directory structure
    try:
        subid = path.parts[-2].split('_')[0]
        if not subid.isnumeric():
            raise RuntimeError(f"Submission id must be numeric, found {subid}")
    except:
        raise RuntimeError(f"Could not parse submission id from '{path}'. File must be in "
                            "directory structure of the form 'TestCase1a/[participant_id]_"
                            "[participant_info]/[submission_id]_[submission_info]/[filename].dat")

    return f'{pid}.{subid}'


# Checks the ONERA OAT15A force and moment submission file
def check_TestCase1a_ForceMoment_file(filename):
    # PurePath can be used to get all parts of the file path
    path = PurePath(filename)

    # check the actual file name
    valid_name = 'DPW8-AePW4_ForceMoment_v5.dat'
    if path.parts[-1] != valid_name:
        raise RuntimeError(f"Filename provided '{path.parts[-1]}' does not match valid file name "
                           f"'{valid_name}'")

    # build the full id from the participant and submission ids
    full_id = get_id_from_path(path)

    # Read the tecplot file and check the data inside
    data = parse_tecplot_file(filename)
    if data['title'] != full_id:
        raise RuntimeError(f"Title inside '{filename}' does not match participant and submission "
                           f"ids, found {data['title']}, expected {full_id}")


# Checks the ONERA OAT15A sectional cuts submission file
def check_TestCase1a_SectionalCuts_file(filename):
    # PurePath can be used to get all parts of the file path
    path = PurePath(filename)

    # check the actual file name
    valid_name = 'DPW8-AePW4_SectionalCuts_v5.dat'
    if path.parts[-1] != valid_name and \
       not re.match(valid_name.split('.dat')[0]+'_ALPHA[0-9].[0-9][0-9]_GRID[0-9]+.dat',path.parts[-1]):
        raise RuntimeError(f"Filename provided '{path.parts[-1]}' does not match valid file names "
                           f"'{valid_name}' or '{valid_name.split('.dat')[0]+'_ALPHA#.##_GRID#.dat'}'")

    # build the full id from the participant and submission ids
    full_id = get_id_from_path(path)

    # Read the tecplot file and check the data inside
    data = parse_tecplot_file(filename)
    if data['title'] != full_id:
        raise RuntimeError(f"Title inside '{filename}' does not match participant and submission "
                           f"ids, found {data['title']}, expected {full_id}")

    # More checks here...
    # potential checks:
    #   - Valid data for required variables like CD, CL, CM


# Checks the ONERA OAT15A convergence submission file
def check_TestCase1a_Convergence_file(filename):
    # PurePath can be used to get all parts of the file path
    path = PurePath(filename)

    # check the actual file name
    valid_name = 'DPW8-AePW4_Convergence_v5.dat'
    if path.parts[-1] != valid_name and \
       not re.match(valid_name.split('.dat')[0]+'_ALPHA[0-9].[0-9][0-9]_GRID[0-9]+',path.parts[-1]):
        raise RuntimeError(f"Filename provided '{path.parts[-1]}' does not match valid file names "
                           f"'{valid_name}' or '{valid_name.split('.dat')[0]+'_ALPHA#.##_GRID#.dat'}'")

    # build the full id from the participant and submission ids
    full_id = get_id_from_path(path)

    # Read the tecplot file and check the data inside
    data = parse_tecplot_file(filename)
    if data['title'] != full_id:
        raise RuntimeError(f"Title inside '{filename}' does not match participant and submission "
                           f"ids, found {data['title']}, expected {full_id}")

    # More checks here...
    # potential checks:
    #   - Valid data for required variables like CD, CL, CM


# Checks file name and spawns additional checks based on the file type
def check_file(filename):
    # ignore .gitignore and markdown files
    if filename.endswith('.md') or 'gitignore' in filename:
        return

    # only .dat files should be in submission folder (other than the .md and .gitignore files)
    if not filename.endswith('.dat'):
        raise RuntimeError(f"Only .dat files can be submitted, found '{filename}'")

    # Check to see if FeCFD or StarkIndustries were included in directory names. These were given
    # as examples in the github instructions.
    if 'starkindustries' in filename.lower():
        raise RuntimeError("StarkIndustries should not be included in your directory structure. "
                           "This was an example organization name, so replace it with your own "
                           "organization.")

    if 'fecfd' in filename.lower():
        raise RuntimeError("FeCFD (Iron CFD) should not be included in your directory structure. "
                           "This was an example solver name, so replace it with your own "
                           "solver.")

    # check which test case this submission belongs to
    if 'TestCase1a' in filename:

        # filename contains the entire path, get the basename for checking
        basename = os.path.basename(filename)
        if 'ForceMoment' in basename:
            check_TestCase1a_ForceMoment_file(filename)
            return

        if 'SectionalCuts' in basename:
            check_TestCase1a_SectionalCuts_file(filename)
            return

        if 'Convergence' in basename:
            check_TestCase1a_Convergence_file(filename)
            return

    raise RuntimeError(f"Filename '{filename}' does not match any accepted filenames for Scatter "
                       "Working Group.")


if __name__ == '__main__':
    print(f'Modified files: {sys.argv[1:]}')
    for filename in sys.argv[1:]:
        print(f'checking {filename}')
        check_file(filename)
