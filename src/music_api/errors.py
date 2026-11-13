from __future__ import annotations


class ApplicationError(Exception):
    def __init__(
        self,
        code: str,
        detail: str,
        status_code: int = 400,
    ) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.status_code = status_code


class ResourceNotFoundError(ApplicationError):
    def __init__(self, detail: str = "The requested resource was not found.") -> None:
        super().__init__("resource_not_found", detail, status_code=404)


class DependencyUnavailableError(ApplicationError):
    def __init__(self, dependency: str) -> None:
        super().__init__(
            "dependency_unavailable",
            f"The {dependency} service is temporarily unavailable.",
            status_code=503,
        )
