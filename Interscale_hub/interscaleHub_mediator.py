# ------------------------------------------------------------------------------
#  Copyright 2020 Forschungszentrum Jülich GmbH
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor
#  license agreements; and to You under the Apache License, Version 2.0. "
#
# Forschungszentrum Jülich
#  Institute: Institute for Advanced Simulation (IAS)
#    Section: Jülich Supercomputing Centre (JSC)
#   Division: High Performance Computing in Neuroscience
# Laboratory: Simulation Laboratory Neuroscience
#       Team: Multi-scale Simulation and Design
# ------------------------------------------------------------------------------ 
from EBRAINS_InterscaleHUB.Interscale_hub.interscalehub_enums import DATA_BUFFER_STATES

from EBRAINS_ConfigManager.global_configurations_manager.xml_parsers.default_directories_enum import DefaultDirectories


class InterscaleHubMediator:
    def __init__(self,  configurations_manager, log_settings,
                 transformer, analyzer, data_buffer_manager):
        self.__logger = configurations_manager.load_log_configurations(
                        name="InterscaleHub -- Mediator",
                        log_configurations=log_settings,
                        target_directory=DefaultDirectories.SIMULATION_RESULTS)

        self.__transformer = transformer
        self.__analyzer = analyzer
        self.__data_buffer_manager = data_buffer_manager

        self.__logger.info("initialized")

    def rate_to_spikes(self):
        '''converts rate to spike trains'''
        if self.__data_buffer_manager.get_at(index=-2) == DATA_BUFFER_STATES.HEADER:
            time_step = self.__data_buffer_manager.get_upto(index=2)
            data_buffer = self.__data_buffer_manager.get_from(starting_index=2)
        else:
            mpi_shared_data_buffer = self.__data_buffer_manager.mpi_shared_memory_buffer
            time_step = self.__data_buffer_manager.get_upto(index=2)
            data_buffer = self.__data_buffer_manager.get_from_range(
                start=2,
                end=int(mpi_shared_data_buffer[-2]))
        
        spike_trains = self.__transformer.rate_to_spikes(time_step, data_buffer)
        self.__logger.debug(f'spikes after conversion: {spike_trains}')
        return spike_trains

    def spikes_to_rate(self, count, size_at_index):
        '''
        Two step conversion from spikes/spike events to firing rates.
        '''
        # TODO refactor buffer indexing and buffer access inside analyzer and transformer
        buffer_size = self.__data_buffer_manager.get_at(index=size_at_index)
        data_buffer = self.__data_buffer_manager.mpi_shared_memory_buffer
        # 1) spike to spike_trains in transformer
        spike_trains = self.__transformer.spike_to_spiketrains(count, buffer_size, data_buffer)
        self.__logger.debug(f'transformed spike trains: {spike_trains}')
        # 2) spike_trains to rate in analyzer
        times, data = self.__analyzer.spiketrains_to_rate(count, spike_trains)
        self.__logger.debug(f'analyzed rates, time: {times}, data: {data}')
        return times, data

    def online_uea_update(self, count, size_at_index, events):
        '''
        1) converts incoming spikes to spiketrains
        NOTE: uses the spike_to_spiketrains method from above

        2) Updates the entries of the result dictionary by processing the
        new arriving 'spiketrains' and trial defining trigger 'events'.
        '''
        # TODO refactor buffer indexing and buffer access inside analyzer and transformer
        buffer_size = self.__data_buffer_manager.get_at(index=size_at_index)
        data_buffer = self.__data_buffer_manager.mpi_shared_memory_buffer
        # 1) spike to spike_trains in transformer
        spike_trains = self.__transformer.spike_to_spiketrains(count, buffer_size, data_buffer)
        # 2) uae with spiketrains in analyzer
        self.__analyzer.online_uea_update(spike_trains, events)
        self.__logger.debug(f'updated uea analysis with spiketrains: {spike_trains}, events: {events}')

    def online_uea_get_results(self):
        '''
        Returns the result dictionary with the following class attribute names
        as keys and the corresponding attribute values as the complementary
        value for the key: (see also Attributes section for respective key
        descriptions)
        * 'Js'
        * 'indices'
        * 'n_emp'
        * 'n_exp'
        * 'rate_avg'
        * 'input_parameters'
        '''
        results = self.__analyzer.online_uea_get_results()
        self.__logger.debug(f'requested results: {results}')
        return results

