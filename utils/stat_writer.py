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
        self.writer = csv.writer(self.file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        self.func = {
            'sum': '=SUMA({})',
            'average': '=MEDIA({})',
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
        for i in range(0, len(header)):
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

    def delete_row(self, row_index):
        assert isinstance(row_index, int), 'row_index is not an integer'
        self.rows.pop(row_index)

    def add_statistic_row(self, row):
        assert isinstance(row, list), 'row is not a list'
        assert len(row) == len(self.header), 'row length is not equal to header length'
        assert self.header is not None, 'header is not defined, please add a column first'
        print('ROW', row)
        self.rows.append(row)
        print('ROWS', self.rows)

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
        pass

    def generate_excel_file(self):
        assert self.header is not None, 'header is not defined, please add a column first'
        assert len(self.rows) > 0, 'rows is empty, please add a row first'

        """table = pd.DataFrame(self.rows, columns=self.header)
        table.to_excel(self.filename + ".xlsx", index=False)"""

        file = open(self.filename + '.csv')
        csvreader = csv.reader(file)
        header = next(csvreader)

        workbook = xlsxwriter.Workbook(self.filename + '.xlsx')
        worksheet1 = workbook.add_worksheet('Male')
        worksheet2 = workbook.add_worksheet('Female')

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
    function = stat_writer.add_function('sum', 2, 1, 3)
    print('function', function)
    stat_writer.add_statistic_row(['', function])
    stat_writer.write_csv_file()
    stat_writer.generate_excel_file()
    stat_writer.empty()
    stat_writer.print()
    stat_writer.read_csv_file()
    stat_writer.print()
