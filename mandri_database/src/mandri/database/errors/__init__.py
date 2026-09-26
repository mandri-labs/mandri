class DatabaseError(Exception):
    pass


class MigrationError(DatabaseError):
    pass


class ConstraintError(DatabaseError):
    pass


class DatabaseConnectionError(DatabaseError):
    pass
