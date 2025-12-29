"""
Scene Flow モジュール

VoteFlowなどのScene Flowモデルを統合するためのモジュール。
"""


def get_voteflow_wrapper():
    """VoteFlowWrapperを遅延インポートして返す"""
    from .voteflow_wrapper import VoteFlowWrapper
    return VoteFlowWrapper


# 後方互換性のため、遅延インポートでVoteFlowWrapperを提供
def __getattr__(name):
    if name == 'VoteFlowWrapper':
        return get_voteflow_wrapper()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ['VoteFlowWrapper', 'get_voteflow_wrapper']

