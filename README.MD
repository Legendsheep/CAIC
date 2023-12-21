The project can be run by running the main_HDC.py file from the src directory CAIC. This is needed to make sure that the paths are properly referenced (to load in enc.csv and weigth.csv).

To generate new weights train_weights.py can be run. (currently implemented with random search, but can be altered to use genetic weight training). The resulting encoding and weights are stored in the enc.csv and weigth.csv.

Each of the testbenches TESTBENCHX.py can be run separetly to make sure that a specific part of the code works and should give the following message if all tests are passed (otherwise an assertionerror is thrown):

passed all tests. Ready for take-off

TESTBENCH3.py has a variable Show_graphs which can be changed to generate a visualisation of the Nelder-Mead steps.

