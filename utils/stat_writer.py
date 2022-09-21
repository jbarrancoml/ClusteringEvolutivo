import csv
import pandas as pd
import xlsxwriter


class StatWriter:
    def __init__(self, filename):
        self.filename = filename
        self.file = open(self.filename + '.csv', "w")

        self.header = None
        self.header_types = list()
        self.rows = list()
        self.statistic_rows = list()

        self.writer = csv.writer(self.file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        self.func = {
            'sum': '=SUMA({})',
            'average': '=PROMEDIO({})',
            'max': '=MAX({})',
            'min': '=MIN({})',
            'count': '=CONTAR({})',
            'std': '=DESVIACION({})',
        }

    def write_header(self, header):
        self.write(header)

    def add_column(self, column, column_type):
        assert isinstance(column, str), 'column is not a string'
        if self.header is None:
            self.header = list()
        self.header.append(column)
        self.header_types.append(column_type)

    def add_header(self, header, header_types):
        assert isinstance(header, list), 'header is not a list'
        assert isinstance(header_types, list), 'header_types is not a list'

        for i in range(0, len(header)):
            assert isinstance(header[i], str), 'header[{}] is not a string'.format(i)
            assert isinstance(header_types[i], type), 'header_types[{}] is not a type'.format(i)

            self.add_column(header[i], header_types[i])

    def add_row(self, row):
        assert isinstance(row, list), 'row is not a list'
        assert len(row) == len(self.header), 'row length is not equal to header length'
        assert self.check_row_types(row), 'row types are not equal to header types'
        assert self.header is not None, 'header is not defined, please add a column first'

        self.rows.append(row)

    def check_row_types(self, row):
        for i in range(0, len(row)):
            if not isinstance(row[i], self.header_types[i]):
                assert False, 'in column {}. Row types are not equal to header types: column type is {} instead of {}'.format(self.header[i], str(self.header_types[i].__name__), str(type(row[i]).__name__))
        return True

    def add_single_column(self, column_name, column_values, column_type, function_cell=None):
        assert isinstance(column_name, str), 'column_name is not a string'
        assert isinstance(column_values, list), 'column_values is not a list'
        assert isinstance(column_type, type), 'column_type is not a type'
        assert len(column_values) == len(self.rows), 'column_values length is not equal to rows length'

        if self.header is None:
            self.header = list()
        self.header.append(column_name)
        self.header_types.append(column_type)

        for i in range(0, len(self.rows)):
            assert column_values[i] is None or isinstance(column_values[i], column_type), 'column_values type is not equal to column_type'
            self.rows[i].append(column_values[i])

        if function_cell is not None:
            assert isinstance(function_cell, str), 'function_cell is not a string'
            self.statistic_rows.append([function_cell])

    def delete_row(self, row_index):
        assert isinstance(row_index, int), 'row_index is not an integer'
        self.rows.pop(row_index)

    def add_statistic_row(self, row):
        assert isinstance(row, list), 'row is not a list'
        assert len(row) == len(self.header), 'row length is not equal to header length'
        assert self.header is not None, 'header is not defined, please add a column first'

        self.statistic_rows.append(row)
        print('ST', self.statistic_rows)

    def get_cell_index(self, column):
        assert isinstance(column, int), 'column is not an integer'
        assert column <= len(self.header), 'column is out of range'

        letters = {
            0: 'A',
            1: 'B',
            2: 'C',
            3: 'D',
            4: 'E',
            5: 'F',
            6: 'G',
            7: 'H',
            8: 'I',
            9: 'J',
            10: 'K',
            11: 'L',
            12: 'M',
            13: 'N',
            14: 'O',
            15: 'P',
            16: 'Q',
            17: 'R',
            18: 'S',
            19: 'T',
            20: 'U',
            21: 'V',
            22: 'W',
            23: 'X',
            24: 'Y',
            25: 'Z',
        }
        return letters[column-1]

    def add_function(self, function, column, first_row=None, last_row=None):
        assert function in self.func, 'function is not defined'
        assert isinstance(first_row, int), 'first_row is not an integer'
        assert isinstance(last_row, int), 'last_row is not an integer'
        assert first_row < last_row, 'first_row is greater than last_row'
        assert first_row > 0, 'first_row is less than 1'
        assert last_row <= len(self.rows), 'last_row is greater than rows length'
        assert (first_row is None and last_row is None) \
               or (first_row is not None and last_row is not None),\
            'first_row and last_row must be both None or both not None'
        assert isinstance(column, int), 'column is not an integer'
        assert column <= len(self.header), 'column is out of range'

        if first_row is None and last_row is None:
            first_row = 2
            last_row = len(self.rows) + 2

        function = self.func[function].format(self.get_cell_index(column) + str(first_row+1) + ':' + self.get_cell_index(column) + str(last_row+1))
        return function

    def print(self):
        print(self.header)
        print(self.rows)

    def close(self):
        self.file.close()

    def write_csv_file(self):
        assert self.header is not None, 'header is not defined, please add a column first'
        assert len(self.rows) > 0, 'rows is empty, please add a row first'

        self.writer.writerow(self.header)
        for row in self.rows:
            self.writer.writerow(row)
        for statictic_row in self.statistic_rows:
            print('statictic_row', statictic_row)
            self.writer.writerow(statictic_row)

    def generate_excel_file(self):
        assert self.header is not None, 'header is not defined, please add a column first'
        assert len(self.rows) > 0, 'rows is empty, please add a row first'

        """table = pd.DataFrame(self.rows + self.statistic_rows, columns=self.header)
        print('Table:', table)
        table.to_excel(self.filename + ".xlsx", index=False)"""

        file = open(self.filename + '.csv')

        workbook = xlsxwriter.Workbook(self.filename + '.xlsx')
        worksheet = workbook.add_worksheet('Statistics')

        # Start from the first cell. Rows and columns are zero indexed.
        row = 0
        col = 0

        # Iterate over the data and write it out row by row.
        for item in self.header:
            worksheet.write(row, col, item)
            col += 1
        col = 0
        row += 1
        for item in self.rows:
            for i in item:
                worksheet.write(row, col, i)
                col += 1
            col = 0
            row += 1
        col = 0
        for item in self.statistic_rows:
            for i in item:
                worksheet.write(row, col, i)
                col += 1
            col = 0
            row += 1

        workbook.close()

        pass

    def read_csv_file(self):
        assert self.filename is not None, 'filename is not defined, please add a filename first'
        table = pd.read_excel(self.filename + '.xlsx')
        self.header = table.columns.values.tolist()
        self.rows = table.values.tolist()
        pass

    def empty(self):
        self.header = list()
        self.header_types = list()
        self.rows = list()

    def __del__(self):
        self.close()


if __name__ == '__main__':
    stat_writer = StatWriter('test')
    stat_writer.add_column('column1', str)
    stat_writer.add_column('column2', int)
    stat_writer.add_row(['row1', 1])
    stat_writer.add_row(['row2', 2])
    stat_writer.add_row(['Row3', 3])
    function = stat_writer.add_function('average', 2, first_row=1, last_row=3)
    stat_writer.add_statistic_row(['', function])
    stat_writer.write_csv_file()
    stat_writer.generate_excel_file()
    stat_writer.print()
