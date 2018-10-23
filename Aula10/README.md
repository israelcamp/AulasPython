## Aula 10 - Funções de ativação

Nesta aula vamos voltar a rede implementada nas aulas [8](https://github.com/israelcamp/AulasPython/tree/master/Aula8) e [9](https://github.com/israelcamp/AulasPython/tree/master/Aula9) e adicionar funções de ativação. Adicionamos ReLU na camada escondida e Sigmoid na camada de saída.

### Outras referências

* [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
* [Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function)


### Observacao

O modo como implementamos a rede nos notebooks nao e o mais eficiente e claro, alem de funcionar somente para datasets com apenas um target.  Logo, escrevi uma arquivo python implementando a mesma rede de uma forma mais generica.