
import dask

from bbb.components import Qlog


from graphree.omni import GraphNode, GraphSchema


# INPUT + 10  -20 -40



ser_json={'anode':{
            "needs":{"startlog":"startvalue"},
          'uses': 'bbb.components.ADD_C',},
'mnode1':{
    "needs": {"anode":'out1'},
          'uses': 'bbb.components.MINUSN_C',
          'config': {"n":20},},
'mnode2':{
    "needs": {"anode":'out1'},
    'uses': 'bbb.components.MINUSN_C',
          'config': {"n":40},},
'pnode':{
    "needs": {"mnode1":"out1",'mnode2':"out1"},
          'uses': 'bbb.components.PROD_C',
          'config': {"cfg":0.01},
          'is_target': True},
}


qlog=Qlog(log={"startlog":("startvalue",90)},runschema=ser_json)

graph_schema=GraphSchema.from_dict(ser_json)
#
run_targets=graph_schema.target_names
run_graph={
            node_name: (GraphNode.from_schema_node(
                node_name, schema_node#可能要多传一个log
            ),*schema_node.needs.keys())
            for node_name, schema_node in graph_schema.nodes.items()
        }
run_graph.update({'startlog':('startlog',('startvalue',[qlog]))})
dask_result = dask.get(run_graph, run_targets)
q=dask_result[0][1][1][0]
print(dask_result[0][1][1][0])
#
#
#
# if __name__ == '__main__':
#
#     def add(a,b):
#         return a+b
#     from dask.threaded import get
#     import random
#
#     TG={'x': [1],
#      'y': [2],
#      'z': (add, 'x', 'y'),#元组即task,第一个元素是函数
#      #'w': (sum, ['x', 'y', 'z']),
#      #'v': [(sum, ['w', 'z']), 2]
#         }
#     get(TG, ['z'])
