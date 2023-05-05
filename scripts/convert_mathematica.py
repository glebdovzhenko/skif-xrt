import re
import random
import string
import os


class Parser:
    fn_translations = {'Abs': 'np.abs', 'Sqrt': 'np.sqrt', 'E': 'np.exp', 'Cos': 'np.cos', 'Sin': 'np.sin', 'Re': 'np.real'}
    op_translations = {'*': '*', '/': '/', '+': '+', '-': '-', '^': '**'}
    
    def __init__(self, var_size=10):
        self.syntax_tree = []
        self.expr_idx = 0 
        self.expr_stub = ''.join(random.choice(string.ascii_uppercase) for _ in range(var_size)) + \
            '%0' + str(var_size) + 'd'

    def __repr__(self):
        return 'Var stub: ' + self.expr_stub + '\nTree:\n' + '\n'.join(map(str, self.syntax_tree))
    
    @staticmethod
    def cleanup(source_str):
        """
        Removes \n and multiple consecutive whitespaces.
        """
        source_str = source_str.replace('\n', '')
        source_str = re.sub(r'[\s]+', ' ', source_str, count=1000)
        return source_str

    def locate_binary_op(self, source_str):
        """
        Locates +-*/^ and appends them to the syntax tree.
        """
        matches = re.findall(r'(?P<ee>(?P<e1>\w+)\s*(?P<op>[\+\-\^\*/\s])\s*(?P<e2>\w+))', source_str)
        matches = list(filter(
            lambda x: (x[1] not in self.fn_translations.keys()) and (x[3] not in self.fn_translations.keys()),
            matches
        ))
    
        for match in matches:
            self.syntax_tree.append({
                'name': self.expr_stub % self.expr_idx,
                'op': match[2] if not match[2].isspace() else '*',
                'args': (match[1], match[3])
            })
            source_str = source_str.replace(match[0], self.syntax_tree[-1]['name'], 1)
            self.expr_idx += 1

        return bool(matches), source_str

    def locate_unary_op(self, source_str):
        """
        Locates unary - and appends them to the syntax tree
        """
        matches = re.findall(r'(?P<ee>\s*-\s*(?P<arg>\w+))', source_str)
        matches = list(filter(
            lambda x: x[1] not in self.fn_translations.keys(),
            matches
        ))

        for match in matches:
            self.syntax_tree.append({
                'name': self.expr_stub % self.expr_idx,
                'op': '-',
                'args': match[1]
            })
            source_str = source_str.replace(match[0], self.syntax_tree[-1]['name'], 1)
            self.expr_idx += 1
        return bool(matches), source_str


    def locate_fn(self, source_str):
        """
        Locates functions defined in fn_translations and appends them to the syntax tree.
        """
        matches = re.findall(
            r'(?P<ee>(?P<fn>' + '|'.join(self.fn_translations.keys()) + r')\[\s*(?P<arg>\w+)\s*\])', 
            source_str
        )
        matches.extend(re.findall(r'(?P<ee>(?P<fn>E)\^\(\s*(?P<arg>\w+)\s*\))', source_str))
        matches.extend(re.findall(r'(?P<ee>\\\[(?P<fn>Sqrt)\]\(\s*(?P<arg>\w+)\s*\))', source_str))
        for match in matches:
            self.syntax_tree.append({
                'name': self.expr_stub % self.expr_idx,
                'op': match[1],
                'args': match[2]
            })
            source_str = source_str.replace(match[0], self.syntax_tree[-1]['name'], 1)
            self.expr_idx += 1
        return bool(matches), source_str

    def locate_par(self, source_str):
        """
        Locates parentheses and appends them to the syntax tree.
        """
        matches = re.findall(r'(?P<ee>\(\s*(?P<arg>\w+)\s*\))', source_str)
        for match in matches:
            self.syntax_tree.append({
                'name': self.expr_stub % self.expr_idx,
                'op': '()',
                'args': match[1]
            })
            source_str = source_str.replace(match[0], self.syntax_tree[-1]['name'], 1)
            self.expr_idx += 1
        return bool(matches), source_str

    def build_syntax_tree(self, source_str):
        cont = True
        while cont:
            c1, source_str = self.locate_binary_op(source_str)
            c2, source_str = self.locate_unary_op(source_str)
            c3, source_str = self.locate_fn(source_str)
            c4, source_str = self.locate_par(source_str)
            cont = c1 or c2 or c3 or c4
        return source_str

    def translate(self, source_str):
        matches = True
        while matches:
            matches = re.findall(
                self.expr_stub.split('%')[0] + r'\d{' + str(int(self.expr_stub.split('%')[1][:-1])) + '}', 
                source_str
            )
            for match in matches:
                edge = filter(lambda x: x['name'] == match, self.syntax_tree).__next__()
            
                if edge['op'] == '-':
                    if isinstance(edge['args'], str):
                        res = '%s%s' % (edge['op'], edge['args'])
                    else:
                        res = '%s %s %s' % (edge['args'][0], edge['op'], edge['args'][1])
                elif edge['op'] in self.op_translations.keys():
                    res = '%s %s %s' % (edge['args'][0], self.op_translations[edge['op']], edge['args'][1])
                elif edge['op'] == '()':
                    res = '(%s)' % edge['args']
                elif edge['op'] in self.fn_translations.keys():
                    res = '%s(%s)' % (self.fn_translations[edge['op']], edge['args'])
                else:
                    raise ValueError(edge['op'])

                source_str = source_str.replace(match, res)
            
        return source_str

    def get_variables(self):
        var_list = sum(
            map(
                lambda x: x['args'] if isinstance(x['args'], tuple) else (x['args'], ), 
                self.syntax_tree
            ), 
            ()
        )
        var_list = list(filter(lambda x: self.expr_stub.split('%')[0] not in x, var_list))
        for ii in reversed(range(len(var_list))):
            try: 
                float(var_list[ii])
            except ValueError:
                pass
            else:
                del var_list[ii]
        return list(sorted(set(var_list)))


if __name__ == '__main__':
    wd = os.path.join(os.getenv('BASE_DIR'), 'datasets', 'bump_eqs')
    surf, surf_x, surf_y, surf_xy = "", "", "", ""
    norm, norm_x, norm_y, norm_xy = "", "", "", ""
    bnorm, bnorm_x, bnorm_y, bnorm_xy = "", "", "", ""

    with open(os.path.join(wd, 'surf.txt'), 'r') as f:
        surf = f.read()
        parser = Parser()
        surf = parser.cleanup(surf)
        surf = parser.build_syntax_tree(surf)   
        surf = parser.translate(surf)
        print('surf.txt')
    with open(os.path.join(wd, 'surfX.txt'), 'r') as f:
        surf_x = f.read()
        parser = Parser()
        surf_x = parser.cleanup(surf_x)
        surf_x = parser.build_syntax_tree(surf_x)   
        surf_x = parser.translate(surf_x)
        print('surfX.txt')
    with open(os.path.join(wd, 'surfY.txt'), 'r') as f:
        surf_y = f.read()
        parser = Parser()
        surf_y = parser.cleanup(surf_y)
        surf_y = parser.build_syntax_tree(surf_y)   
        surf_y = parser.translate(surf_y)
        print('surfY.txt')
    with open(os.path.join(wd, 'surfXY.txt'), 'r') as f:
        surf_xy = f.read()
        parser = Parser()
        surf_xy = parser.cleanup(surf_xy)
        surf_xy = parser.build_syntax_tree(surf_xy)   
        surf_xy = parser.translate(surf_xy)
        print('surfXY.txt')
    with open(os.path.join(wd, 'norm.txt'), 'r') as f:
        norm = f.read()
        parser = Parser()
        norm = parser.cleanup(norm)
        norm = parser.build_syntax_tree(norm)   
        norm = parser.translate(norm)
        print('norm.txt')
    with open(os.path.join(wd, 'normX.txt'), 'r') as f:
        norm_x = f.read()
        parser = Parser()
        norm_x = parser.cleanup(norm_x)
        norm_x = parser.build_syntax_tree(norm_x)   
        norm_x = parser.translate(norm_x)
        print('normX.txt')
    with open(os.path.join(wd, 'normY.txt'), 'r') as f:
        norm_y = f.read()
        parser = Parser()
        norm_y = parser.cleanup(norm_y)
        norm_y = parser.build_syntax_tree(norm_y)   
        norm_y = parser.translate(norm_y)
        print('normY.txt')
    with open(os.path.join(wd, 'normXY.txt'), 'r') as f:
        norm_xy = f.read()
        parser = Parser()
        norm_xy = parser.cleanup(norm_xy)
        norm_xy = parser.build_syntax_tree(norm_xy)   
        norm_xy = parser.translate(norm_xy)
        print('normXY.txt')
    with open(os.path.join(wd, 'bnorm.txt'), 'r') as f:
        bnorm = f.read()
        parser = Parser()
        bnorm = parser.cleanup(bnorm)
        bnorm = parser.build_syntax_tree(bnorm)   
        bnorm = parser.translate(bnorm)
        print('bnorm.txt')
    with open(os.path.join(wd, 'bnormX.txt'), 'r') as f:
        bnorm_x = f.read()
        parser = Parser()
        bnorm_x = parser.cleanup(bnorm_x)
        bnorm_x = parser.build_syntax_tree(bnorm_x)   
        bnorm_x = parser.translate(bnorm_x)
        print('bnormX.txt')
    with open(os.path.join(wd, 'bnormY.txt'), 'r') as f:
        bnorm_y = f.read()
        bnorm_y = parser.cleanup(bnorm_y)
        bnorm_y = parser.build_syntax_tree(bnorm_y)   
        bnorm_y = parser.translate(bnorm_y)
        print('bnormY.txt')
    with open(os.path.join(wd, 'bnormXY.txt'), 'r') as f:
        bnorm_xy = f.read()
        bnorm_xy = parser.cleanup(bnorm_xy)
        bnorm_xy = parser.build_syntax_tree(bnorm_xy)   
        bnorm_xy = parser.translate(bnorm_xy)
        print('bnormXY.txt')

    with open(os.path.join(os.getenv('BASE_DIR'), 'components', 'bump_eqs.py'), 'w') as f:
    # with open('test.py', 'w') as f:
        f.write('import numpy as np\n')
               # Rx, Ry finite
        f.write('def crs(xS, yS, Cx, Cy, Rx, Ry, Sx, Sy, Axy):\n')
        f.write('    return ' + surf + '\n\n\n')
        f.write('def crn(xS, yS, Cx, Cy, Rx, Ry, Sx, Sy, Axy):\n')
        f.write('    return ' + norm.replace('{', '[').replace('}', ']') + '\n\n\n')
        f.write('def crbn(xS, yS, Cx, Cy, Rx, Ry, Sx, Sy, Axy, Chi):\n')
        f.write('    return ' + bnorm.replace('{', '[').replace('}', ']') + '\n\n\n')
        # Ry is infinite 
        f.write('def crs_x(xS, yS, Cx, Cy, Rx, Ry, Sx, Sy, Axy):\n')
        f.write('    return ' + surf_x + '\n\n\n')
        f.write('def crn_x(xS, yS, Cx, Cy, Rx, Ry, Sx, Sy, Axy):\n')
        f.write('    return ' + norm_x.replace('{', '[').replace('}', ']') + '\n\n\n')
        f.write('def crbn_x(xS, yS, Cx, Cy, Rx, Ry, Sx, Sy, Axy, Chi):\n')
        f.write('    return ' + bnorm_x.replace('{', '[').replace('}', ']') + '\n\n\n')
        # Rx is infinite
        f.write('def crs_y(xS, yS, Cx, Cy, Rx, Ry, Sx, Sy, Axy):\n')
        f.write('    return ' + surf_y + '\n\n\n')
        f.write('def crn_y(xS, yS, Cx, Cy, Rx, Ry, Sx, Sy, Axy):\n')
        f.write('    return ' + norm_y.replace('{', '[').replace('}', ']') + '\n\n\n')
        f.write('def crbn_y(xS, yS, Cx, Cy, Rx, Ry, Sx, Sy, Axy, Chi):\n')
        f.write('    return ' + bnorm_y.replace('{', '[').replace('}', ']') + '\n\n\n')
        # Rx, Ry are infinite
        f.write('def crs_xy(xS, yS, Cx, Cy, Rx, Ry, Sx, Sy, Axy):\n')
        f.write('    return ' + surf_xy + '\n\n\n')
        f.write('def crn_xy(xS, yS, Cx, Cy, Rx, Ry, Sx, Sy, Axy):\n')
        f.write('    return ' + norm_xy.replace('{', '[').replace('}', ']') + '\n\n\n')
        f.write('def crbn_xy(xS, yS, Cx, Cy, Rx, Ry, Sx, Sy, Axy, Chi):\n')
        f.write('    return ' + bnorm_xy.replace('{', '[').replace('}', ']') + '\n\n\n')

        f.write('if __name__ == \'__main__\':')
        f.write("""
    from matplotlib import pyplot as plt

    x, y, z = np.meshgrid(np.linspace(-5, 5, 15), np.linspace(-5, 5, 15), [0.])
    args = [x, y, 0., 0., -10., 10., 1., 1., 2., np.radians(35.3)]

    z = crs(*args[:-1])
    u1, v1, w1 = crn(*args[:-1])
    u2, v2, w2 = crbn(*args)
    ax = plt.figure().add_subplot(projection='3d')      
    ax.plot_surface(x[:, :, 0], y[:, :, 0], z[:, :, 0])
    ax.quiver(x, y, z, u1, v1, w1, normalize=False, color='r')
    ax.quiver(x, y, z, u2, v2, w2, normalize=False, color='g')
    plt.xlabel('local x')
    plt.ylabel('local y')
    ax.set_xlim(-5., 5.)
    ax.set_ylim(-5., 5.)
    ax.set_zlim(0., 10.)

    z = crs_x(*args[:-1])
    u1, v1, w1 = crn_x(*args[:-1])
    u2, v2, w2 = crbn_x(*args)   
    ax = plt.figure().add_subplot(projection='3d')      
    ax.plot_surface(x[:, :, 0], y[:, :, 0], z[:, :, 0])
    ax.quiver(x, y, z, u1, v1, w1, normalize=False, color='r')
    ax.quiver(x, y, z, u2, v2, w2, normalize=False, color='g')
    plt.xlabel('local x')
    plt.ylabel('local y')
    ax.set_xlim(-5., 5.)
    ax.set_ylim(-5., 5.)
    ax.set_zlim(0., 10.)

    z = crs_y(*args[:-1])
    u1, v1, w1 = crn_y(*args[:-1])
    u2, v2, w2 = crbn_y(*args)      
    ax = plt.figure().add_subplot(projection='3d')      
    ax.plot_surface(x[:, :, 0], y[:, :, 0], z[:, :, 0])
    ax.quiver(x, y, z, u1, v1, w1, normalize=False, color='r')
    ax.quiver(x, y, z, u2, v2, w2, normalize=False, color='g')
    plt.xlabel('local x')
    plt.ylabel('local y')
    ax.set_xlim(-5., 5.)
    ax.set_ylim(-5., 5.)
    ax.set_zlim(0., 10.)

    z = crs_xy(*args[:-1])
    u1, v1, w1 = crn_xy(*args[:-1])
    u2, v2, w2 = crbn_xy(*args)   
    ax = plt.figure().add_subplot(projection='3d')      
    ax.plot_surface(x[:, :, 0], y[:, :, 0], z[:, :, 0])
    ax.quiver(x, y, z, u1, v1, w1, normalize=False, color='r')
    ax.quiver(x, y, z, u2, v2, w2, normalize=False, color='g')

    plt.xlabel('local x')
    plt.ylabel('local y')
    ax.set_xlim(-5., 5.)
    ax.set_ylim(-5., 5.)
    ax.set_zlim(0., 10.)
    plt.show()
    """)

