#include <stdio.h>
#include "pico/stdlib.h"
#include "hardware/i2c.h"
#include "lib/include/ssd1306.h"               // Biblioteca para display OLED SSD1306
#include "tflm_wrapper.h"
#include "wine_dataset.h"          // Contém wine_features[178][13] e wine_labels[178]
#include "wine_normalization.h"    // Contém os vetores wine_means[13] e wine_stds[13]


#define NUM_SAMPLES 178            // Total de amostras no dataset Wine
#define NUM_CLASSES 3              // Wine possui 3 classes (0, 1 e 2)

#define BUTTON_A 5
#define I2C_PORT_DISPLAY i2c1
#define I2C_SDA_DISPLAY 14
#define I2C_SCL_DISPLAY 15
#define ADDRESS_DISPLAY 0x3C

ssd1306_t disp;

// Matriz de confusão 3x3: real x predito
static int conf_matrix[NUM_CLASSES][NUM_CLASSES];

/*
 * Função normalize_input
 * Aplica a normalização padrão (StandardScaler):
 *     x_norm = (x - mean) / std
 * Essa normalização precisa ser a mesma usada no treinamento em Python.
 */
void normalize_input(const float in[13], float out[13]) {
    for (int i = 0; i < 13; i++)
        out[i] = (in[i] - wine_means[i]) / wine_stds[i];
}

/*
 * Função argmax
 * Retorna o índice da maior probabilidade entre scores[3]
 * É equivalente ao np.argmax() do Python.
 */
int argmax(const float v[3]) {
    int idx = 0;
    float max = v[0];
    for (int i = 1; i < 3; i++)
        if (v[i] > max) { max = v[i]; idx = i; }
    return idx;
}

void wait_for_button_press() {
    while (gpio_get(BUTTON_A) == 1) {
        ssd1306_draw_string(&disp, "Pressione A", 30, 15);
        ssd1306_draw_string(&disp, "para Inferir", 28, 35);
        ssd1306_send_data(&disp);
        tight_loop_contents();
    }
    
    sleep_ms(200); // debounce simples
}

// Função principal
int main() {

    // Inicializa printf via USB no Pico W
    stdio_init_all();

    gpio_init(BUTTON_A);
    gpio_set_dir(BUTTON_A, GPIO_IN);
    gpio_pull_up(BUTTON_A);

    i2c_init(I2C_PORT_DISPLAY, 400 * 1000);
    gpio_set_function(I2C_SDA_DISPLAY, GPIO_FUNC_I2C);
    gpio_set_function(I2C_SCL_DISPLAY, GPIO_FUNC_I2C);
    gpio_pull_up(I2C_SDA_DISPLAY);
    gpio_pull_up(I2C_SCL_DISPLAY);

    ssd1306_init(&disp, 128, 64, false, ADDRESS_DISPLAY, I2C_PORT_DISPLAY);
    ssd1306_config(&disp);

    printf("\n=== TinyML Wine - Matriz de Confusao ===\n");

    
    // Inicializa o TensorFlow Lite Micro (intérprete, arena, tensores etc.)
    if (tflm_init_model() != 0) {
        printf("Falha ao inicializar modelo.\n");
        return 1;
    }

    // 🔵 Espera o botão
    wait_for_button_press();

    printf("Modelo inicializado com sucesso!\n");
    printf("Iniciando inferencia nas 178 amostras do dataset Wine...\n");


    // Zera matriz de confusão 3×3
    for (int i = 0; i < NUM_CLASSES; i++)
        for (int j = 0; j < NUM_CLASSES; j++)
            conf_matrix[i][j] = 0;

    int correct = 0;      // Acertos totais

    /*
     * Loop principal de inferência.
     * Percorre todas as 178 amostras do dataset Wine.
     */
    for (int i = 0; i < NUM_SAMPLES; i++) {
        float features_norm[13];    // Entrada normalizada
        float scores[3];           // Saída do modelo (probabilidades para 3 classes)
        
        // Normaliza a entrada usando média e desvio salvos
        normalize_input(wine_features[i], features_norm);

        // Executa inferência via wrapper TFLM
        tflm_infer(features_norm, scores);

        // Classe predita pelo modelo
        int pred = argmax(scores);

        // Classe verdadeira do dataset
        int real = wine_labels[i];

        // Conta acertos
        if (pred == real) correct++;

        // Atualiza matriz de confusão
        conf_matrix[real][pred]++;

        // Exibe apenas as 15 primeiras previsões, para inspeção
        if (i < 15) {
            printf("Amostra %3d  Real: %d  Pred: %d  [%.3f %.3f %.3f]\n",
                i, real, pred, scores[0], scores[1], scores[2]);
        }
    }

    // Exibe matriz de confusão formatada
    printf("\nMatriz de Confusao (real vs predito)\n");
    printf("            Pred0         Pred1       Pred2\n");
    for (int r = 0; r < NUM_CLASSES; r++) {
        printf("Real %d", r);
        for (int c = 0; c < NUM_CLASSES; c++) {
            printf("   %8d", conf_matrix[r][c]);
        }
        printf("\n");
    }

    // Calcula acurácia final
    float accuracy = (float)correct / NUM_SAMPLES;
    printf("\nAcuracia final: %.4f  ( %d / %d )\n",
            accuracy, correct, NUM_SAMPLES);


    ssd1306_fill(&disp, false);

    // Dimensões da tabela
    int cell_w  = 35;
    int cell_h  = 18;
    int start_x = 0;
    int start_y = 10;

    ssd1306_draw_string(&disp, "Matriz Confusao", 5, 0);

    ssd1306_line(&disp,
        start_x,
        start_y + 0 * cell_h,
        start_x + cell_w * NUM_CLASSES,
        start_y + 0 * cell_h,
        true);

    ssd1306_line(&disp,
        start_x,
        start_y + 1 * cell_h,
        start_x + cell_w * NUM_CLASSES,
        start_y + 1 * cell_h,
        true);

    ssd1306_line(&disp,
        start_x,
        start_y + 2 * cell_h,
        start_x + cell_w * NUM_CLASSES,
        start_y + 2 * cell_h,
        true);

    ssd1306_line(&disp,
        start_x,
        start_y + 3 * cell_h - 3,
        start_x + cell_w * NUM_CLASSES,
        start_y + 3 * cell_h - 3,
        true);


    int y_end = start_y + cell_h * NUM_CLASSES - 3;

    ssd1306_line(&disp,
        start_x + 0 * cell_w,
        start_y,
        start_x + 0 * cell_w,
        y_end,
        true);

    ssd1306_line(&disp,
        start_x + 1 * cell_w,
        start_y,
        start_x + 1 * cell_w,
        y_end,
        true);

    ssd1306_line(&disp,
        start_x + 2 * cell_w,
        start_y,
        start_x + 2 * cell_w,
        y_end,
        true);

    ssd1306_line(&disp,
        start_x + 3 * cell_w,
        start_y,
        start_x + 3 * cell_w,
        y_end,
        true);


    // --- Valores nas células ---
    for (int r = 0; r < NUM_CLASSES; r++) {
        for (int c = 0; c < NUM_CLASSES; c++) {

            char buf[6];
            sprintf(buf, "%d", conf_matrix[r][c]);

            int text_x = start_x + c * cell_w + 8;
            int text_y = start_y + r * cell_h + 6;

            ssd1306_draw_string(&disp, buf, text_x, text_y);
        }
    }

    ssd1306_send_data(&disp);


    printf("\nFim da inferencia. Loop infinito.\n");

    // Loop infinito - evita reset do programa
    while(1) tight_loop_contents();
}
