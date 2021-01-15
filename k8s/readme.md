## Configure Kubernetes
1. Install Docker Desktop and activate Kubernetes
2. Install linkerd:
3. Install prometheus via helm with modified values (including linkerd scrape config):
   
```` helm install --values prometheus_values.yaml prometheus prometheus-community/kube-prometheus-stack````