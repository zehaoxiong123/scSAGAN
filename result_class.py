
class result_for_impute():
    def __init__(self,function_name, mat, data_label,cell_type,cell_type_num, obs, var):  # 所有self跟着的参数都是这个类的实例属性
        self.function_name = function_name
        self.data_mat = mat
        self.data_label = data_label
        self.cell_type = cell_type
        self.cell_type_num = cell_type_num
        self.gene_name = var
        self.cell_name = obs
