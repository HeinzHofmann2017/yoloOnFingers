# -*- coding: utf-8 -*-

import argparse
class make_parser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--name',
                            nargs = '?',
                            const = 'DesktopModell',
                            default = 'DesktopModell',
                            help = "Name, which is used to save Models, summarys, etc.",
                            type=str)
                            
        parser.add_argument('--learningrate',
                            nargs = '?',
                            const = 0.001,
                            default = 0.001,
                            help = "learningrate",
                            type=float)
        
        parser.add_argument('--batchSize',
                            nargs = '?',
                            const = 2,
                            default = 2,
                            help = "batchSize",
                            type=int)
                            
        parser.add_argument('--numThreads',
                            nargs = '?',
                            const = 16,
                            default = 16,
                            help = "Number of Threads used to read in Pictures",
                            type=int)
                            
        parser.add_argument('--bufferSize',
                            nargs = '?',
                            const = 20,
                            default = 20,
                            help = "size of the Buffer, within the Data will be shuffled, can easy be about 100'000",
                            type=int)
                            
        parser.add_argument('--nrOfEpochs',
                            nargs = '?',
                            const = 3,
                            default = 3,
                            help = "Nr. of Epochs until stop training",
                            type=int)
                            
        parser.add_argument('--nrOfEpochsUntilSaveModel',
                            nargs = '?',
                            const = 1,
                            default = 1,
                            help = "after this number of Epochs, the model will be saved, validatet and checked",
                            type=int)
        
        parser.add_argument('--originPath',
                            nargs = '?',
                            const = "/home/hhofmann/Schreibtisch/data_hhofmann/Data/indexfinger_right/3000_readyTOlearn/trainData/",
                            default = "/home/hhofmann/Schreibtisch/data_hhofmann/Data/indexfinger_right/3000_readyTOlearn/trainData/",
                            help = "Path to the list with all Files, this is the origin-path, from it everything will be handled",
                            type=str)
        parser.add_argument('--noDropout',
                            dest='dropout',
                            action='store_false',
                            help = "If flag --noDropout is activated, Dropout wont be used.")
        parser.set_defaults(dropout=True)
        
        parser.add_argument('--noBatchnorm',
                            dest='batchnorm',
                            action='store_false',
                            help = "If flag --noBatchnorm is activated, batchnorm wont be used.")
        parser.set_defaults(batchnorm=True)
        
        parser.add_argument('--Test',
                            dest='Test',
                            action='store_true',
                            help = "If Flag Test is activatet, there will be a test applied to the Validation Set and Saved anywhere...")
        parser.set_defaults(batchnorm=False)
                                    
        args = parser.parse_args()
        self.modelname                      = args.name
        self.learning_rate                  = args.learningrate
        self.batch_Size                     = args.batchSize
        self.num_Threads                    = args.numThreads
        self.buffer_Size                    = args.bufferSize
        self.nr_of_epochs                   = args.nrOfEpochs
        self.nr_of_epochs_until_save_model  = args.nrOfEpochsUntilSaveModel        
        self.origin_Path                    = args.originPath
        self.dropout_bool                   = args.dropout
        self.batchnorm_bool                 = args.batchnorm
        self.test_bool                      = args.Test

        
        
        

        
            
if __name__ == "__main__":
   
    lokal_parser    = make_parser()
    print("batchSize        = "+ str(lokal_parser.batch_Size))
    print("learningrate     = "+ str(lokal_parser.learning_rate))
    print("numThreads       = "+ str(lokal_parser.num_Threads))
    print("bufferSize       = "+ str(lokal_parser.buffer_Size))
    print("originPath       = "+ str(lokal_parser.origin_Path))
    print("nrOfEpochs       = "+ str(lokal_parser.nr_of_epochs))
    print("nrOfEpUntilSave  = "+ str(lokal_parser.nr_of_epochs_until_save_model))    
    print("Modelname        = "+ str(lokal_parser.modelname))
    print("Dropout_bool     = "+ str(lokal_parser.dropout_bool))
    print("Batchnorm_bool   = "+ str(lokal_parser.batchnorm_bool))
    pritn("test_bool        = "+ str(lokal_parser.test_bool))