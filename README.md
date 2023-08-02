# RPC tutorial

> The name in this tutorial: `client`, `server`, `worker`, all belongs to `node` class.

## rpc is quite similar to ros2, am I right?

Yes, in some ways, PyTorch's Remote Procedure Call (RPC) framework is similar to ROS2 (Robot Operating System 2) in that they both allow for distributed computing and interprocess communication.

1. **Inter-Process Communication:** Both RPC and ROS2 facilitate inter-process communication, which is the exchange of data across multiple and potentially distributed processes.

2. **Remote Procedure Calls:** RPC and ROS2 both support the idea of remote procedure calls, allowing a process to invoke a procedure or method in another process either on the same machine or another machine on the network. In ROS2, this is facilitated through services, while in PyTorch, it's done through the RPC API.

3. **Asynchronous Communication:** Both systems provide support for both synchronous and asynchronous communication. In PyTorch, `rpc_sync` and `rpc_async` handle this, while ROS2 provides similar functionality with services (synchronous) and topics (asynchronous).

However, it's also important to note that the two systems have been developed for very different purposes. PyTorch RPC is primarily meant for distributed deep learning, while ROS2 is designed for the needs of complex robotic systems. This results in some key differences in their architecture and usage. For instance, ROS2 has a more comprehensive system for message passing, with well-defined message types, topics, and services, and it supports real-time communication, which is crucial in robotics applications.