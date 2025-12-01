import matplotlib.pyplot as plt
import numpy as np

def draw_neural_network(hidden_layers, neurons_per_layer, activation_name):
    """
    Рисует архитектуру нейронной сети с использованием Matplotlib.
    """
    # Конфигурация слоев: [Вход (1), ...Скрытые..., Выход (1)]
    layer_sizes = [1] + [neurons_per_layer] * hidden_layers + [1]

    # Параметры отрисовки
    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca()
    ax.axis('off')

    left, right, bottom, top = 0.1, 0.9, 0.1, 0.9
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(len(layer_sizes) - 1)

    # Хранилище координат нейронов для отрисовки связей
    # nodes[layer_index] = [ (x, y), (x, y), ... ]
    nodes = []

    # 1. Расчет координат узлов (Nodes)
    for l, n in enumerate(layer_sizes):
        layer_nodes = []
        layer_top = v_spacing * (n - 1) / 2.0 + (top + bottom) / 2.0
        for i in range(n):
            x = left + l * h_spacing
            y = layer_top - i * v_spacing
            layer_nodes.append((x, y))
        nodes.append(layer_nodes)

    # 2. Отрисовка связей (Edges)
    # Рисуем линии ПЕРЕД кружками, чтобы они были на заднем плане
    for l in range(len(layer_sizes) - 1):
        for i, (x1, y1) in enumerate(nodes[l]):
            for j, (x2, y2) in enumerate(nodes[l + 1]):
                # Прозрачность линий зависит от количества нейронов (чтобы не было каши)
                alpha = 0.3 if neurons_per_layer < 15 else 0.1
                line = plt.Line2D([x1, x2], [y1, y2], c='#999999', alpha=alpha, linewidth=1, zorder=1)
                ax.add_artist(line)

    # 3. Отрисовка нейронов и подписей
    node_radius = 0.03 if neurons_per_layer < 20 else 0.015

    for l, layer_nodes in enumerate(nodes):
        # Подписи слоев
        x_center = layer_nodes[0][0]
        if l == 0:
            ax.text(x_center, top + 0.05, "Input", ha='center', fontsize=12, fontweight='bold')
        elif l == len(layer_sizes) - 1:
            ax.text(x_center, top + 0.05, "Output", ha='center', fontsize=12, fontweight='bold')
        else:
            ax.text(x_center, top + 0.05, f"Hidden {l}", ha='center', fontsize=10)

        # Отрисовка самих кружков
        for i, (x, y) in enumerate(layer_nodes):
            circle = plt.Circle((x, y), node_radius, color='w', ec='#007acc', linewidth=2, zorder=2)
            ax.add_artist(circle)

            # Если слоев мало, подпишем bias (условно) или индекс
            if neurons_per_layer <= 5:
                ax.text(x, y, "", ha='center', va='center', fontsize=6)

    # 4. Отрисовка функции активации (Стрелки и текст между слоями)
    for l in range(len(layer_sizes) - 2): # -2, так как после последнего скрытого слоя к выходу активации (обычно) нет или она identity
        # Координаты середины между слоями
        x_curr = nodes[l+1][0][0]
        x_next = nodes[l+2][0][0] # На самом деле активация идет ПОСЛЕ скрытого слоя

        # Текст активации пишем над слоем l+1
        # Но чтобы было понятнее, напишем между связями

        mid_x = x_curr + 0.1 # Слегка сдвигаем вправо от скрытого слоя

        # Просто добавим аннотацию внизу или сверху диаграммы
        ax.text(x_curr, bottom - 0.05, f"Activation:\n{activation_name}",
                ha='center', va='top', fontsize=9, color='darkred', style='italic')

    # Общая подпись
    ax.set_title(f"Network Architecture: {hidden_layers} hidden layer(s)", fontsize=10)

    return fig