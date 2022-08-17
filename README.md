# About
This is a support vector regression (SVR) Kubernetes autoscaler which was developed during my masters thesis. It is able to scale horizontally and vertically based on historical data.

# Usage
First Kubernetes needs to be setup. Then the loadtesting can be started.  After that the generated raw data can be filtered.
Then the machine learning algorithms can be trained. Finally, the trained models can be used in the autoscaler.
The description of the used methods can be found in the code documentation.
The dataset as well as the trained machine learning models that are used in the thesis can be found in the archive called `teastore final.zip` in the `datasets` folder of the repository.

## Project Setup

 1. Open the git bash
 2. Clone the repository e.g. via https: 
	 `git clone https://git.rwth-aachen.de/parallel/sci-staff/dissemination.git`
 3.  Go to the `app` directory.
 4. Open a Python IDE of your choice (e.g. [PyCharm](https://www.jetbrains.com/de-de/pycharm/))
 5. Create a new virtual Environment using [Python 3.8](https://www.python.org/downloads/release/python-387/)
 6. Install the required packages via pip: 
	 `pip install -r requirements.txt`
	 

## Kubernetes Setup

1.  Install [Docker Desktop](https://www.docker.com/products/docker-desktop) and activate Kubernetes
    
2.  Install [linkerd v2.9](https://linkerd.io/2.9/getting-started/):
    
    2a. Download v2.9 of linkerd from the [release page](https://github.com/linkerd/linkerd2/releases/tag/stable-2.9.5)
    
    2b. extract it and copy it to `$HOME`
    
    2c. Export path:
    
    `export PATH=$PATH:$HOME/.linkerd2/bin`
    
    2d. Install:
    
    `linkerd install | kubectl apply -f -`
    
    2d. Wait until finished:
    
    `linkerd check`
    
3.  Install [Prometheus](https://prometheus.io/) via [helm](https://helm.sh/) with modified values (including linkerd scrape config):
    
    `cd k8s`
    
    `helm install --values prometheus_values.yaml prometheus prometheus-community/kube-prometheus-stack`
    
4.  Change ClusterIP to NordPort for both Prometheus services:
    
    4a. Prometheus installed via helm:
    
    `kubectl patch svc prometheus-kube-prometheus-prometheus --type='json' -p '[{"op":"replace","path":"/spec/type","value":"NodePort"}]'`
    
    4b. Prometheus installed via linkerd:
    
    `kubectl patch svc -n linkerd linkerd-prometheus --type='json' -p '[{"op":"replace","path":"/spec/type","value":"NodePort"}]'`

## Loadtesting
You can choose to use the GUI or the python script itself. The functionallity of the GUI is described in the thesis in detail (simply start the `gui.py` file). Here the usage of the python script is described:
 1. Open the `benchmark.py` file in a python IDE.
 2. Start the `start` method. The raw data will be saved in the folder `data/raw/`.
 
## Filtering/Formatting
The raw data from laodtesting has to be filtered to be used as a dataset for the machine learning models and formatted to be used for the tool [Extra-P](https://github.com/extra-p/extrap):
 1. Open the `formatting.py` file in a python IDE.
 2. Start the `process_all_runs` method.
 3. The formatted data for the dataset will be saved in `data/filtered`, while the data formatted for Extra-P will be saved in `data/formatted`.
 
 ## Train the Machine Learning Models
 The hyperparameters of each machine learning method can be tweaked in each of their corresponding methods. In order to train the machine learning models with the created dataset the following steps have to be executed:
 
  1. Open the `ml.py` file in a python IDE.
  2. Start the ``processes_data`` method to split the dataset into a training and a test dataset.
  3. Start the ``train_for_all_targets(kind: str)`` method with the variable `kind` being the desired machine learning model. It can be chosen from "neural" ([MLPRegressor NN](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html?highlight=mlpregressor#sklearn.neural_network.MLPRegressor)), "linear" ([Bayesian Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html?highlight=bayesian%20linear#sklearn.linear_model.BayesianRidge)) and "svr" ([Support Vector Regression](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html?highlight=svr#sklearn.svm.SVR)).
  4. The trained models are then saved in `data/models`.

## Deploy the Autoscaler
To use the autoscaler the coresponding Docker image has to be configurated, build and pushed.

 1. Open the `benchmark.py` file with a python IDE.
 2. Use the ``change_build`` method to configure and build the Docker image (current name:`angi2412/autoscaler`). The name of the Docker image can be changed in the ``build_autoscaler_docker()`` method of the `k8s_tools.py` file.
 3. Push the build image to a registry of your choice (check that you are logged in to this registry): 
	 `docker push angi2412/autoscaler`
 4. Deploy the autoscaler with the build Docker image to the Kubernetes Cluster (change the docker image name in the yaml file if necessary): 
	 `kubectl apply -f k8s/autoscaler.yaml` 

## Evaluation Runs
If you want to recreate the evaluation runs, you have to first deploy the autoscaler like described above and then execute the following steps:

 1. Open the `benchmark.py` file with a python IDE.
 2. Use the `evaluation` method with the desired parameters.
 3. The data of the evaluation runs will also be saved in the `data/raw` folder and can be filtered analogous to the raw dataset data.
 4. The evaluation metrics and plots can be created with the ``plot_all_evaluation``  method of the `formatting.py`file.

# Citation
Horn, A., Fard, H.M., Wolf, F. (2022). Multi-objective Hybrid Autoscaling of Microservices in Kubernetes Clusters. In: Cano, J., Trinder, P. (eds) Euro-Par 2022: Parallel Processing. Euro-Par 2022. Lecture Notes in Computer Science, vol 13440. Springer, Cham. https://doi.org/10.1007/978-3-031-12597-3_15

BibTex:
```
@InProceedings{10.1007/978-3-031-12597-3_15,
author="Horn, Angelina
and Fard, Hamid Mohammadi
and Wolf, Felix",
editor="Cano, Jos{\'e}
and Trinder, Phil",
title="Multi-objective Hybrid Autoscaling of Microservices in Kubernetes Clusters",
booktitle="Euro-Par 2022: Parallel Processing",
year="2022",
publisher="Springer International Publishing",
address="Cham",
pages="233--250",
abstract="The cloud community has accepted microservices as the dominant architecture for implementing cloud native applications. To efficiently execute microservice-based applications, application owners need to carefully scale the required resources, considering the dynamic workload of individual microservices. The complexity of resource provisioning for such applications highlights the crucial role of autoscaling mechanisms. Kubernetes, the common orchestration framework for microservice-based applications, mainly proposes a horizontal pod autoscaling (HPA) mechanism, which, however, lacks efficiency. To hinder resource wastage and still achieve the requested average response time of microservices, we propose a multi-objective autoscaling mechanism. Based on machine learning techniques, we introduce a toolchain for hybrid autoscaling of microservices in Kubernetes. Comparing several machine learning techniques and also our in-house performance modeling tool, called Extra-P, we propose the most adequate model for solving the problem. Our extensive evaluation on a real-world benchmark application shows a significant reduction of resource consumption while still meeting the average response time specified by the user, which outperforms the results of common HPA in Kubernetes.",
isbn="978-3-031-12597-3"
}
```
