from collections import OrderedDict

num_query=15
#######################################
# Base超参设置
#######################################
config = OrderedDict()
config['dataset_name'] = 'CropDisease'
config['num_generation'] = 15
config['num_loss_generation'] = 3
config['generation_weight'] = 0.5
config['light_distance_metric'] = 'l1'
config['color_distance_metric'] = 'l1'
config['emb_size'] = 128
config['backbone'] = 'labnet'


#######################################
# 测试阶段超参
#######################################
test_opt = OrderedDict()
test_opt['num_ways'] = 5
test_opt['num_shots'] = 1
test_opt['num_queries'] = num_query      
test_opt['batch_size'] = 4   
test_opt['iteration'] = 5000  
test_opt['loss_indicator'] = [1, 1, 1, 0] 
test_opt['dropout'] = 0.1


###################################
config['test_config'] = test_opt
###################################