import os
import urllib
import candle
from data_utils import candle_data_dict, Downloader

CANDLE_DATA_DIR=os.getenv("CANDLE_DATA_DIR")

file_path = os.path.dirname(os.path.realpath(__file__))
additional_definitions = [
    {'name': 'data_type',
     'type': str
     },
    {'name': 'data_url',
     'type': str
     },
    {'name': 'data_split_seed',
     'type': int
     },
     {'name': 'metric',
     'type': str
     },
     {'name': 'data_version',
     'type': str,
     'help': '....'
     }
]

required = None


class Paccmann_candle(candle.Benchmark):

        def set_locals(self):
            if required is not None:
                self.required = set(required)
            if additional_definitions is not None:
                self.additional_definitions = additional_definitions

def initialize_parameters():
    """ Initialize the parameters for the GraphDRP benchmark. """
    print("Initializing parameters\n")
    swnet_params = Paccmann_candle(
        filepath = file_path,
        defmodel = "paccmannmca_default_model.txt",
        framework = "pytorch",
        prog="Paccmann",
        desc="CANDLE compliant Paccmann",
    )
    gParameters = candle.finalize_parameters(swnet_params)
    return gParameters

if __name__ == '__main__':

    params = initialize_parameters()
    data_type = candle_data_dict[params['data_source']]
    split_id = params['data_split_id']
    data_version = params['data_version']

    # data_url = gParameters['data_url']

    data_path = os.path.join(CANDLE_DATA_DIR, params['model_name'], 'Data')
    downloader = Downloader(data_version)
    downloader.download_candle_data(data_type=data_type, split_id=split_id, data_dest=data_path)
    # download_candle_data(data_type=data_type, split_id=0, data_dest=data_path)


    fname='Data_MCA.zip'
    origin=params['data_url']
    # Download and unpack the data in CANDLE_DATA_DIR
    candle.file_utils.get_file(fname=fname, 
    origin=origin, 
    cache_subdir='Paccmann_MCA')


