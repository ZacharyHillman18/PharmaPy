# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 14:49:31 2023

@author: dcasasor
"""

import json
import pathlib

from PharmaPy.Errors import PharmaPySpecificationError
import warnings


root = pathlib.Path(__file__).parent / 'data'


def check_modeling_objects(uo, instance_name=None):
    class_name = uo.__class__.__name__

    # A ConcurrentVessel holds no modeling objects of its own: every unit it
    # integrates does, so each of those is checked under a qualified name.
    members = getattr(uo, 'concurrent_members', None)

    if members:
        prefix = class_name if instance_name is None else instance_name

        for member_name, member in members.items():
            check_modeling_objects(member, '%s.%s' % (prefix, member_name))

        return

    with open(root / 'minimum_modeling_objects.json') as fi:
        checks = json.load(fi)

    if instance_name is None:
        instance_name = '<InstanceName>'
        instance_descr = "a " + class_name + ' instance'
    else:
        instance_descr = "the '%s' %s instance" % (instance_name, class_name)

    if class_name in checks['special']:
        modeling_objs = checks['special'][class_name]
    else:
        modeling_objs = checks[uo.oper_mode]

    module_name = uo.__module__.split('.')[-1]

    cond_kin = (module_name in checks['has_kinetics']['modules'] or
                class_name in checks['has_kinetics']['classes']) and \
        'Kinetics' not in modeling_objs

    cond_utility = (module_name in checks['has_utility']['modules'] or
                    class_name in checks['has_utility']['classes']) and \
        'Utility' not in modeling_objs

    if cond_kin:
        modeling_objs.append('Kinetics')

    if cond_utility:
        modeling_objs.append('Utility')

    missing_obj = []
    for obj in modeling_objs:
        # Vessels built on MultiPhaseVessel store their feeds in
        # inlet_connections; their Inlet is a write-only convenience setter,
        # so reading it always raises and hasattr alone reports it missing.
        if obj == 'Inlet' and getattr(uo, 'inlet_connections', None):
            continue

        if not hasattr(uo, obj) or getattr(uo, obj) is None:
            missing_obj.append(obj)

    if len(missing_obj) > 0:
        intro = "The following PharmaPy modeling objects were " \
            "not detected in %s:\n" % instance_descr

        obj_enum = '\t' + ',\n\t'.join(missing_obj) + '.\n\n'

        recommend = "Please create the missing modeling objects listed above" \
            " and then aggregate them one by one to the corresponding unit " \
            "operation instance, e.g. %s.%s = <%sClass>(...)" % (
                    instance_name, missing_obj[0], missing_obj[0])

        message = intro + obj_enum + recommend

        # raise PharmaPySpecificationError(message)
        warnings.warn(message)
