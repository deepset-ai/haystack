# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from canals import component


class Remainder:
    @staticmethod
    def create(divisor: int = 3):
        """
        Redirects the value, unchanged, along the connection corresponding to the remainder
        of a division. For example, if `divisor=3`, the value `5` would be sent along
        the second output connection.
        """

        @component
        class RemainderImpl:
            """
            Implementation of Reminder()
            """

            __name__ = __qualname__ = f"Remainder_{divisor}"

            def __init__(self, divisor):
                if divisor == 0:
                    raise ValueError("Can't divide by zero")
                self.divisor = divisor

            @component.return_types(**{f"remainder_is_{val}": int for val in range(divisor)})
            def run(self, value: int):
                """
                :param value: the value to check the remainder of.
                """
                remainder = value % self.divisor
                output = {f"remainder_is_{val}": None if val != remainder else value for val in range(self.divisor)}
                return output

        return RemainderImpl(divisor=divisor)

    def __init__(self):
        raise NotImplementedError("use Remainder.create()")
