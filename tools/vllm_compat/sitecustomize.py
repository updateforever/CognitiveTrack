"""vLLM 0.11 与新版 Starlette 的可观测性兼容补丁。

Starlette 0.52 引入的 ``_IncludedRouter`` 没有 ``path`` 属性，而
prometheus-fastapi-instrumentator 7.1 仍假定所有路由对象都有该属性，导致
任何 vLLM HTTP 请求在 metrics 中间件内返回 500。这个问题不涉及模型服务；
metrics 只需要一个稳定的低基数 handler 标签，因此直接使用请求路径即可。

仅在启动 vLLM 时把本目录放到 ``PYTHONPATH`` 最前面。不要把它加入项目的全局
Python 路径，以免影响普通训练与评测进程。
"""

try:
    from prometheus_fastapi_instrumentator import routing

    def _request_path(request):
        return str(request.scope.get("path", "unknown"))

    routing.get_route_name = _request_path
except ImportError:
    # 非 vLLM 进程可能没有该可选依赖；sitecustomize 不应阻断解释器启动。
    pass
