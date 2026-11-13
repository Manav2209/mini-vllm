

class Request:
    def __init__(
        self,
        request_id: str,
        prompt_tokens: list[int],
    ):
        #Your implementation 
        self.request_id = request_id
        self.prompt_len = len(prompt_tokens)
        self.token_ids = list(prompt_tokens)

        self.num_computed_tokens = 0
        self.block_ids: list[int] = []
        

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)


    @property 
    def output_tokens(self) -> list[int]:
        return self.token_ids[self.prompt_len:]


    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)
