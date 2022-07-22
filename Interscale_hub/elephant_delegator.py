# ------------------------------------------------------------------------------
#  Copyright 2020 Forschungszentrum Jülich GmbH and Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements; and to You under the Apache License,
# Version 2.0. "
#
# Forschungszentrum Jülich
# Institute: Institute for Advanced Simulation (IAS)
# Section: Jülich Supercomputing Centre (JSC)
# Division: High Performance Computing in Neuroscience
# Laboratory: Simulation Laboratory Neuroscience
# Team: Multi-scale Simulation and Design
# ------------------------------------------------------------------------------
from EBRAINS_InterscaleHUB.Interscale_hub.delegation.elephant_plugin import ElephantPlugin
from EBRAINS_InterscaleHUB.Interscale_hub.delefation.online_uea import OnlineUnitaryEventAnalysis
from EBRAINS_InterscaleHUB.Interscale_hub.delegation.spike_rate_inter_conversion import SpikeRateConvertor

from EBRAINS_ConfigManager.global_configurations_manager.xml_parsers.default_directories_enum import DefaultDirectories


class ElephantDelegator:
    """
    NOTE: some functionalities only had on attribute/method, e.g. rate_to_spike.
    -> new Class "spike_rate_conversion" contains all related functionalities.
    """
    def __init__(self, param, configurations_manager, log_settings, sci_params=None):
        """

        """
        self._log_settings = log_settings
        self._configurations_manager = configurations_manager
        self.__logger = self._configurations_manager.load_log_configurations(
                                        name="ElephantDelegator",
                                        log_configurations=self._log_settings,
                                        target_directory=DefaultDirectories.SIMULATION_RESULTS)
        # init members
        self.spike_rate_conversion = SpikeRateConvertor(
                                        param, 
                                        configurations_manager, 
                                        log_settings,
                                        sci_params=sci_params)
        self.elephant_plugin = ElephantPlugin(
                                        configurations_manager, 
                                        log_settings)
        # TODO proper wrapping in ElephantPlugin?
        # TODO init with trigger_events (param known beforehand?)
        # and/or pass trigger events at runtime
        # TODO trigger events with Quantities --> move conversion to plugin
        self.online_uea = OnlineUnitaryEventAnalysis()

        # dir member methods
        self.spikerate_methods = [f for f in dir(SpikeRateConvertor) if not f.startswith('_')]
        self.plugin_methods = [f for f in dir(ElephantPlugin) if not f.startswith('_')]
        self.uea_methods = [f for f in dir(OnlineUnitaryEventAnalysis) if not f.startswith('_')]
        self.__logger.info("Initialised")

    def __getattr__(self, func):
        """
        """
        # TODO add support to access the attributes of the classes to be delegated
        def method(*args):
            if func in self.spikerate_methods:
                return getattr(self.spike_rate_conversion, func)(*args)
            elif func in self.plugin_methods:
                return getattr(self.elephant_plugin, func)(*args)
            elif func in self.uea_methods:
                return getattr(self.online_uea, func)(*args)
            else:
                raise AttributeError
        return method
