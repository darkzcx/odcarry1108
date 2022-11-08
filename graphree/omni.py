import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Text, Type, Tuple

from dood.oddo import class_from_module_path


class GraphComponent(ABC):
    """Interface for any component which will run in a graph."""

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        **kwargs: Any,
    ) :...

    def process(self,List)->List:
        ...


@dataclass
class SchemaNode:
    needs:Dict[Text, Text]#一般K表类型，V是节点名
    uses: Type[GraphComponent]#class
    config: Dict[Text, Any]
    is_target: bool = False


#本质就是target_names和nodes
@dataclass
class GraphSchema:
    """Represents a graph for training a model or making predictions."""

    nodes: Dict[Text, SchemaNode]
    @property
    def target_names(self) -> List[Text]:
        """Returns the names of all target nodes."""
        return [node_name for node_name, node in self.nodes.items() if node.is_target]
    def as_dict(self) -> Dict[Text, Any]:
        serializable_graph_schema: Dict[Text, Dict[Text, Any]] = {"nodes": {}}
        for node_name, node in self.nodes.items():
            serializable = dataclasses.asdict(node)

            # Classes are not JSON serializable (surprise)
            serializable["uses"] = f"{node.uses.__module__}.{node.uses.__name__}"

            serializable_graph_schema["nodes"][node_name] = serializable

        return serializable_graph_schema



    @classmethod
    def from_dict(cls, serialized_graph_schema: Dict[Text, Any]) :

        nodes = {}
        for node_name, serialized_node in serialized_graph_schema.items():
            serialized_node[
                "uses"
            ] = class_from_module_path(
                serialized_node["uses"])
            if serialized_node.get('config',None):
                serialized_node['config'].update({'identifier':node_name})
            else:
                serialized_node['config']={'identifier': node_name}
            nodes[node_name] = SchemaNode(**serialized_node)

        return GraphSchema(nodes)


#gs=GraphSchema.from_dict(ser_json)
class GraphNode:
    """
    用于计算图的节点
    """
    def __init__(
        self,
        node_name: Text,
        component_class: Type[GraphComponent],
        #constructor_name: Text,
        component_config: Dict[Text, Any],
        #fn_name: Text,
        inputs: Dict[Text, Text],
    ) -> None:

        self._node_name: Text = node_name
        self._component_class: Type[GraphComponent] = component_class
        self._constructor_name: Text = 'load'
        self._constructor_fn: Callable = getattr(
            self._component_class, self._constructor_name
        )
        self._component_config: Dict[Text, Any] = component_config
        self._fn_name: Text = 'process'
        self._fn: Callable = getattr(self._component_class, self._fn_name)
        #output = self._fn(self._component, **run_kwargs)是可以直接运算的

        self._inputs: Dict[Text, Text] = inputs

        self._component: Optional[GraphComponent] = None

        self._load_component()

    def _load_component(self, **kwargs: Any) -> None:

        constructor = getattr(self._component_class, self._constructor_name)
        try:
            self._component: GraphComponent = constructor(  # type: ignore[no-redef]
                config=self._component_config,
                **kwargs,
            )
        except Exception:
            # Pass through somewhat expected exception to allow more fine granular
            # handling of exceptions.
            raise
        except Exception as e:
            raise Exception(
                f"Error initializing graph component for node {self._node_name}."
            ) from e


    def __call__(
        self, *inputs_from_previous_nodes: Tuple[Text, Any]
    ) -> Tuple[Text, Any]:
        #传入的参数KV
        #received_inputs: Dict[Text, Any] = dict(inputs_from_previous_nodes)

        kwargs = {}
        #_inputs是参数会传入，即schema_node.needs
        #本质是对接了一下用用needs的k  中间是node的 key接上node的value
        # for input_name, input_node in self._inputs.items():
        #     kwargs[input_name] = received_inputs[input_node]

        try:
            #_fn是_component_class的_fn_name函数（通常process,非静态）
            #_component是_component_class的_constructor_name函数通常是load静态构造的
            #构造后本质直接用了call
            #因为直接调用所以需要传self进去，也就是构造后的对象
            output = self._fn(self._component,qlogs=inputs_from_previous_nodes[0][1][1])# **kwargs#kwargs[input_name]
            #本质也是UM这里输出
        except Exception:
            raise


        return self._node_name, output#自己的名字和输出[]


    @classmethod
    def from_schema_node(
        cls,
        node_name: Text,
        schema_node: SchemaNode,
    ) :
        """Creates a `GraphNode` from a `SchemaNode`."""
        return cls(
            node_name=node_name,
            component_class=schema_node.uses,
            component_config=schema_node.config,
            inputs=schema_node.needs,
        )
