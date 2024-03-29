# Configure Kubernetes
1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop) and activate Kubernetes
2. Install [linkerd](https://linkerd.io):
    
    2a. Download: 
   
    ```curl -sL https://run.linkerd.io/install | sh```
    
    2b. Export path: 
    
    ```export PATH=$PATH:$HOME/.linkerd2/bin```

    2c. Install:
    
    ```linkerd install | kubectl apply -f -```

    2d. Wait until finished:
   
    ```linkerd check```
   
3. Install [Prometheus](https://prometheus.io/) via [helm](https://helm.sh/) with modified values (including linkerd scrape config):
   
   ````cd k8s````
   
   ````helm repo add prometheus-community https://prometheus-community.github.io/helm-charts````
   
   ````helm repo update````
   
   ```` helm install --values prometheus_values.yaml prometheus prometheus-community/kube-prometheus-stack````

4. Change ClusterIP to NordPort for both Prometheus services:
   
   4a. Prometheus installed via helm:
   
   ````kubectl patch svc prometheus-kube-prometheus-prometheus --type='json' -p '[{"op":"replace","path":"/spec/type","value":"NodePort"}]'````

   4b. Prometheus installed via linkerd:
   
   ````kubectl patch svc -n linkerd linkerd-prometheus --type='json' -p '[{"op":"replace","path":"/spec/type","value":"NodePort"}]'````
