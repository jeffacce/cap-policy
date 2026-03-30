import zmq
import time
import pickle
import threading
import traceback


class RPCServer:
    def __init__(self, obj, port: int = 5000, threaded=False):
        """
        obj: object with methods to expose
        port: port to listen on
        """
        self.obj = obj
        self.context = zmq.Context()
        self.socket: zmq.socket.Socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        self.threaded = threaded
        if threaded:
            self.thread = threading.Thread(target=self.run)
            self.stop_event = threading.Event()
        else:
            self.stop_event = False

    def _send_exception(self, e):
        """
        Serialize an exception and send it over the socket.
        Only the exception type, message, and traceback are sent.
        """
        exception = {
            "type": "exception",
            "content": {
                "exception": str(type(e)),
                "message": str(e),
                "traceback": traceback.format_exc(),
            },
        }
        self.socket.send(pickle.dumps(exception))

    def _send_result(self, result):
        """
        Serialize a result and send it over the socket.
        """
        result = {"type": "result", "content": result}
        self.socket.send(pickle.dumps(result))

    def run(self):
        """
        Run the server.
        """
        if self.threaded:
            while not self.stop_event.is_set():
                message = self.socket.recv()
                message = pickle.loads(message)
                self._handle_message(message)
        else:
            while not self.stop_event:
                try:
                    message = self.socket.recv(flags=zmq.NOBLOCK)
                    message = pickle.loads(message)
                except zmq.Again:
                    time.sleep(0.001)
                    continue
                self._handle_message(message)

    def _is_callable(self, attr):
        return hasattr(self.obj, attr) and callable(getattr(self.obj, attr))

    def _handle_message(self, message):
        """
        Handles a dictionary of {
            "req": str,  # request type
            "attr": str,
            "args": list,
            "kwargs": dict,
        }
        from the socket.
        If req == "is_callable", return whether the attribute is callable.
        If req == "get", return the attribute.
            If the attribute is not found, return an error message.
            If the attribute is callable, call with args and kwargs.
                If there are any errors in the callable, return the pickled error
                If the callable is found and there are no errors, return the pickled result.
            If the attribute is not callable, return the attribute.
        If req == "set", set the attribute to the value.
        If req == "dir", return a list of attributes.
        If req == "stop", stop the server.
        """
        if message["req"] == "is_callable":
            result = self._is_callable(message["attr"])
            self._send_result(result)
        elif message["req"] == "get":
            try:
                attribute = getattr(self.obj, message["attr"])
                args = message["args"]
                kwargs = message["kwargs"]
                if not callable(attribute):
                    self._send_result(attribute)
                else:
                    result = attribute(*args, **kwargs)
                    self._send_result(result)
            except Exception as e:
                self._send_exception(e)
        elif message["req"] == "set":
            try:
                setattr(self.obj, message["attr"], message["value"])
                self._send_result(None)
            except Exception as e:
                self._send_exception(e)
        elif message["req"] == "dir":
            result = dir(self.obj)
            self._send_result(result)
        elif message["req"] == "stop":
            self.stop()

    def close(self):
        self.socket.close()
        self.context.term()

    def start(self):
        if self.threaded:
            self.stop_event.clear()
            self.thread.start()
        else:
            self.run()

    def stop(self):
        if self.threaded:
            self.stop_event.set()
            self.thread.join()
        else:
            self.stop_event = True
        self.close()


class RPCException(Exception):
    def __init__(self, exception_type: str, message: str, traceback: str):
        self.exception_type = exception_type
        self.message = message
        self.traceback = traceback

    def __str__(self):
        return f"{self.exception_type}: {self.message}\n{self.traceback}"


class RPCClient:
    def __init__(self, host: str, port: int = 5000):
        """
        host: host to connect to
        port: port to connect to
        """
        self.__dict__["context"] = zmq.Context()
        self.__dict__["socket"] = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{host}:{port}")
        self.__dict__["_is_callable_cache"] = {}

    def __setattr__(self, attr: str, value):
        """
        Set the attribute of the same name.
        Attribute must not be callable on remote.
        """
        if self._is_callable(attr):
            raise AttributeError(f"Overwriting a callable attribute: {attr}")
        self._send_set(attr, value)

    def _send_get(self, attr: str, args: list, kwargs: dict):
        """
        Send a get request over the socket.
        """
        req = {"req": "get", "attr": attr, "args": args, "kwargs": kwargs}
        self.socket.send(pickle.dumps(req))
        return self._recv_result()

    def _send_set(self, attr: str, value):
        """
        Send a set request over the socket.
        """
        req = {"req": "set", "attr": attr, "value": value}
        self.socket.send(pickle.dumps(req))
        return self._recv_result()

    def _recv_result(self):
        """
        Receive a dictionary of {
            "type": str,
            "content": object,
        }
        if type == "exception", content is a dictionary of {
            "exception": str,
            "message": str,
            "traceback": str,
        }; re-raise the exception on the client side
        if type == "result", content is the result
        """
        result = self.socket.recv()
        result = pickle.loads(result)
        if result["type"] == "exception":
            raise RPCException(
                result["content"]["exception"],
                result["content"]["message"],
                result["content"]["traceback"],
            )
        return result["content"]

    def _is_callable(self, attr: str) -> bool:
        """
        Send a request to check if the attribute is callable.
        Returns False if the attribute is not found.
        """
        if attr not in self._is_callable_cache:
            req = {"req": "is_callable", "attr": attr}
            self.socket.send(pickle.dumps(req))
            result = self._recv_result()
            self._is_callable_cache[attr] = result
        return self._is_callable_cache[attr]

    def __getattr__(self, attr: str):
        """
        Return the attribute of the same name.
        If the attribute is a callable, return a function that sends the call over the socket.
        Else, return the attribute value.
        """
        if self._is_callable(attr):
            return lambda *args, **kwargs: self._send_get(attr, args, kwargs)
        else:
            return self._send_get(attr, [], {})

    def __dir__(self):
        """
        Return a list of attributes.
        """
        req = {"req": "dir"}
        self.socket.send(pickle.dumps(req))
        result = self._recv_result()
        return result + ["stop_server"]

    def stop_server(self) -> bool:
        """
        Send a stop request to the server.
        If the server is stopped, close the socket and terminate the context.
        Returns a bool for success.
        """
        req = {"req": "stop"}
        self.socket.send(pickle.dumps(req))
        stopped = self._recv_result()
        if stopped:
            self.socket.close()
            self.context.term()
        return stopped
