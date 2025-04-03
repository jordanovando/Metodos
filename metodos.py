import tkinter as tk
from tkinter import messagebox
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class RootFinderApp:
    def __init__(self, master):
        self.master = master
        master.title("Métodos Numéricos - Secante y Newton-Raphson")
        
        # Configuración de la interfaz
        self.setup_ui()
        
    def setup_ui(self):
        """Configura todos los elementos de la interfaz gráfica"""
        main_frame = tk.Frame(self.master)
        main_frame.pack(padx=10, pady=10)
        
        # Entrada de función
        tk.Label(main_frame, text="Función f(x):").grid(row=0, column=0, sticky="w")
        self.func_entry = tk.Entry(main_frame, width=40)
        self.func_entry.grid(row=0, column=1, padx=5, pady=5)
        self.func_entry.insert(0, "x**3 - 2*x - 5")  # Ejemplo por defecto
        
        # Selector de método
        tk.Label(main_frame, text="Método:").grid(row=1, column=0, sticky="w")
        self.method_var = tk.StringVar(value="secante")
        tk.Radiobutton(main_frame, text="Secante", variable=self.method_var, value="secante").grid(row=1, column=1, sticky="w")
        tk.Radiobutton(main_frame, text="Newton-Raphson", variable=self.method_var, value="newton").grid(row=2, column=1, sticky="w")
        
        # Valores iniciales
        tk.Label(main_frame, text="Valor inicial 1 (x0):").grid(row=3, column=0, sticky="w")
        self.x0_entry = tk.Entry(main_frame, width=15)
        self.x0_entry.grid(row=3, column=1, sticky="w", padx=5, pady=5)
        self.x0_entry.insert(0, "2.0")
        
        tk.Label(main_frame, text="Valor inicial 2 (x1 - solo Secante):").grid(row=4, column=0, sticky="w")
        self.x1_entry = tk.Entry(main_frame, width=15)
        self.x1_entry.grid(row=4, column=1, sticky="w", padx=5, pady=5)
        self.x1_entry.insert(0, "3.0")
        
        # Parámetros numéricos
        tk.Label(main_frame, text="Tolerancia:").grid(row=5, column=0, sticky="w")
        self.tol_entry = tk.Entry(main_frame, width=15)
        self.tol_entry.insert(0, "1e-6")
        self.tol_entry.grid(row=5, column=1, sticky="w", padx=5, pady=5)
        
        tk.Label(main_frame, text="Máx. iteraciones:").grid(row=6, column=0, sticky="w")
        self.max_iter_entry = tk.Entry(main_frame, width=15)
        self.max_iter_entry.insert(0, "100")
        self.max_iter_entry.grid(row=6, column=1, sticky="w", padx=5, pady=5)
        
        # Botones
        self.calc_button = tk.Button(main_frame, text="Calcular Raíz", command=self.calculate_root)
        self.calc_button.grid(row=7, column=0, columnspan=2, pady=10)
        
        # Área de resultados
        self.result_text = tk.Text(main_frame, height=10, width=60, state="normal")
        self.result_scroll = tk.Scrollbar(main_frame, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=self.result_scroll.set)
        self.result_text.grid(row=8, column=0, columnspan=2, pady=5)
        self.result_scroll.grid(row=8, column=2, sticky="ns")
        
        # Gráficas
        self.figure, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 4))
        self.canvas = FigureCanvasTkAgg(self.figure, master=main_frame)
        self.canvas.get_tk_widget().grid(row=9, column=0, columnspan=3, pady=10)
    
    def calculate_root(self):
        """Control principal para calcular la raíz"""
        try:
            # Validar y obtener datos de entrada
            f_expr, f, x0, x1, tol, max_iter = self.validate_input()
            
            # Ejecutar método seleccionado
            if self.method_var.get() == "secante":
                root, iterations, errors = self.secant_method(f, x0, x1, tol, max_iter)
            else:
                root, iterations, errors = self.newton_method(f_expr, x0, tol, max_iter)
            
            # Mostrar resultados
            self.display_results(root, iterations, errors)
            self.plot_results(f, iterations, errors, root)
            
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error:\n{str(e)}")
    
    def validate_input(self):
        """Valida todos los campos de entrada"""
        try:
            # Validar función matemática
            x = sp.symbols('x')
            f_expr = sp.sympify(self.func_entry.get())
            f = sp.lambdify(x, f_expr, 'numpy')
            
            # Validar valores numéricos
            x0 = float(self.x0_entry.get())
            x1 = float(self.x1_entry.get()) if self.method_var.get() == "secante" else None
            tol = float(self.tol_entry.get())
            max_iter = int(self.max_iter_entry.get())
            
            # Validación específica por método
            if self.method_var.get() == "secante":
                if x1 is None:
                    raise ValueError("El método Secante requiere dos valores iniciales")
                if abs(f(x0) - f(x1)) < 1e-12:
                    raise ValueError("f(x0) y f(x1) no deben ser iguales")
            
            return f_expr, f, x0, x1, tol, max_iter
            
        except sp.SympifyError:
            raise ValueError("Expresión matemática no válida")
        except ValueError as e:
            raise ValueError(f"Dato numérico inválido: {str(e)}")
    
    def secant_method(self, f, x0, x1, tol, max_iter):
        """Implementación del método de la secante"""
        iterations = [x0, x1]
        errors = []
        
        for i in range(max_iter):
            fx0 = f(x0)
            fx1 = f(x1)
            
            if abs(fx1 - fx0) < 1e-20:
                raise ValueError("Diferencia entre f(x1) y f(x0) demasiado pequeña")
                
            x_new = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
            error = abs(x_new - x1)
            
            iterations.append(x_new)
            errors.append(error)
            
            if error < tol:
                break
                
            x0, x1 = x1, x_new
        
        return x1, iterations[2:], errors
    
    def newton_method(self, f_expr, x0, tol, max_iter):
        """Implementación del método de Newton-Raphson"""
        x = sp.symbols('x')
        f = sp.lambdify(x, f_expr, 'numpy')
        df_expr = sp.diff(f_expr, x)
        df = sp.lambdify(x, df_expr, 'numpy')
        
        iterations = [x0]
        errors = []
        
        for i in range(max_iter):
            fx = f(x0)
            dfx = df(x0)
            
            if abs(dfx) < 1e-20:
                raise ValueError("Derivada cercana a cero")
                
            x_new = x0 - fx / dfx
            error = abs(x_new - x0)
            
            iterations.append(x_new)
            errors.append(error)
            
            if error < tol:
                break
                
            x0 = x_new
        
        return x0, iterations[1:], errors
    
    def display_results(self, root, iterations, errors):
        """Muestra los resultados en el área de texto"""
        self.result_text.config(state="normal")
        self.result_text.delete(1.0, tk.END)
        
        self.result_text.insert(tk.END, f"Raíz aproximada: {root:.10f}\n")
        self.result_text.insert(tk.END, f"Número de iteraciones: {len(iterations)}\n")
        self.result_text.insert(tk.END, f"Error final: {errors[-1]:.2e}\n\n")
        self.result_text.insert(tk.END, "Detalle de iteraciones:\n")
        self.result_text.insert(tk.END, "-"*60 + "\n")
        self.result_text.insert(tk.END, "Iter\tx\t\t\tError\n")
        self.result_text.insert(tk.END, "-"*60 + "\n")
        
        for i, (xi, err) in enumerate(zip(iterations, errors)):
            self.result_text.insert(tk.END, f"{i+1:3d}\t{xi:.10f}\t{err:.2e}\n")
        
        self.result_text.config(state="disabled")
    
    def plot_results(self, f, iterations, errors, root):
        """Genera las gráficas de resultados"""
        self.ax1.clear()
        self.ax2.clear()
        
        # Gráfica de la función y aproximaciones
        x_min = min(iterations) - 1
        x_max = max(iterations) + 1
        x_vals = np.linspace(x_min, x_max, 400)
        y_vals = f(x_vals)
        
        self.ax1.plot(x_vals, y_vals, label='f(x)')
        self.ax1.axhline(0, color='black', linewidth=0.5)
        self.ax1.scatter(iterations, [0]*len(iterations), color='red', label='Aproximaciones')
        self.ax1.scatter([root], [0], color='green', marker='x', s=100, label='Raíz encontrada')
        self.ax1.set_title('Función y aproximaciones')
        self.ax1.set_xlabel('x')
        self.ax1.set_ylabel('f(x)')
        self.ax1.legend()
        self.ax1.grid(True)
        
        # Gráfica de convergencia del error
        self.ax2.semilogy(range(1, len(errors)+1), errors, 'o-')
        self.ax2.set_title('Convergencia del error')
        self.ax2.set_xlabel('Iteración')
        self.ax2.set_ylabel('Error (log)')
        self.ax2.grid(True)
        
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = RootFinderApp(root)
    root.mainloop()