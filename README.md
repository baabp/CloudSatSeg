# CloudSatSeg
### setup environment
#### start docker container
```
$ docker-compose start
$ docker exec -it baabp-cloud_master bash
```

or 

```
docker-compose run -p 8009:8009 --name=baabp-cloud_master_run baabp-cloud_master bash
```
??

### Jupyter
* Settings on docker
```
# jupyter notebook --generate-config
# jupyter notebook password
```

* launch
```
# jupyter lab --ip=0.0.0.0 --port=8008 --allow-root
```

### tensorboard
```
# tensorboard --logdir runs --port 6006 --host 0.0.0.0
```

### tips

#### torchsummary
```
from torchsummary import summary
summary(model, (1, 28, 28))
```

## References
### 