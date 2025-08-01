custom_error_codes = {
    # File reading errors
    1000: {"message": "Invalid file, path doesn't exists"},
    1001: {"message": "Invalid file, expected TIFF file with suffix .tif or .tiff"},
    1002: {"message": "Invalid file, expected LSM file with suffix .lsm"},
    1003: {"message": "Invalid channel passed, must be higher or equal to 0"},
    1004: {
        "message": "Invalid channel of interest passed, must be a number  lower to the number of channels"
    },
    1005: {"message": "Invalid file, the file passed isn't supported"},
}


class PycrogliaException(Exception):
    def __init__(self, error_code: int):
        super().__init__(f"Error Code {error_code} : {custom_error_codes[error_code]}")

        self.error_code = error_code
