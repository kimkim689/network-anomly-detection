## Prerequisites
Python 3.8 or higher 
Requirement packages 

## Installation
To install the system, the user can either download the file .zip packet or clone the repository

* Download Source File: 
1. Navigate to the link: https://github.com/kimkim689/network-anomly-detection 
2. Find the latest version tag
3. Download the Source code zip file 
4. Extract the zip 
5. When use, user must cd to the directory containing the file package 
6. Download the Required Package 
```
pip install -r requirements.txt 
```
* Clone Repository
1. Clone the repo using the following command: 
```
git clone https://github.com/kimkim689/network-anomly-detection

```
2. Change directory to the system file: 
```cd network-anomal-detection``` 
3. Install Required Package 
```pip install -r requirements.txt ```

# System Training - Retraining 
Before any analysis, the model must be trained to produce a joblib file:
```
python -m system_main.main --mode train --model [rf,xgboost]  or python -m system_main.main --mode train to train both model at the same time 
```
Random Forest model takes from 30s to 1 minute to train. XGBoost may take up to 3 minutes. When training, keep the device running until training finishes. 

# Pcap Analyser 
```
python -m system_main.main --mode detect_pcap --model [rf,xgboost, combine] --input [path_to_your_pcap_file.pcap]
```
# Live Network Analyser 

* List of network interfaces: 
```
python -m system_main.main --mode list_interfaces
```
* Analyser
```
python -m system_main.main --mode detect_live --model [rf,xgboost]--interface "[your_network_interface]"
```
** remember to double quote the interface name **
* Report Generation 
For detailed report: add “--detail” to the end of the command. For example: 
```
python -m system_main.main --mode detect_pcap --model [rf,xgboost,combine] --input [path_to_your_pcap_file.pcap] –detail
```


any enquiry please contact - bui.k.nguyen@student.uts.edu.au
