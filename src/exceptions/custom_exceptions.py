class FinDocBaseException(Exception):
    """Base exception for all project errors."""

    def __init__(self, message: str, detail: str | None = None):
        self.detail = detail
        super().__init__(message)


class PDFProcessingError(FinDocBaseException):
    """Raised when PDF text/image extraction fails."""
    pass


class VectorStoreError(FinDocBaseException):
    """Raised when FAISS indexing or retrieval fails."""
    pass


class AgentError(FinDocBaseException):
    """Raised when an agent (summary/chart/comparison) fails."""
    pass


class OrchestratorError(FinDocBaseException):
    """Raised when the orchestrator cannot route a query."""
    pass


class StockPriceError(FinDocBaseException):
    """Raised when the Yahoo Finance tool fails."""
    pass


class SessionNotFoundError(FinDocBaseException):
    """Raised when a requested session_id does not exist."""
    pass
