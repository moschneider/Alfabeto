
import java.awt.Color;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.*;

import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JPanel;

public class ANN {

	final int GENEREC = 0;
	final int BACKPROPAGATION = 1;
	
	/* REDE NEURAL ARTIFICIAL (PERCEPTRON DE 3 CAMADAS)
	 * -------------------------------------------------
	 * 
	 * A = numero de entradas
	 * B = numero de neuronios escondidos
	 * C = numero de saidas
	 * 
	 * x[i] = entradas (i = 0,..,A-1)
	 * h[j] = ativacoes na camada escondida (j = 0,..,B-1)
	 * o[k] = saidas calculadas (k = 0,..,C-1)
	 * y[k] = saidas desejadas (k = 0,..,C-1)
	 * 
	 * w[i][j] = sinapses entre camada de entrada e camada escondida (i = 0,..,A-1; j = 0,..,B-1)
	 * q[j][k] = sinapses entre camada escondida e camada de saida (j = 0,..,B-1; k = 0,..,C-1)
	 * 
	 */
	
	int A, B, C;
	
	double x[], h[], o[], y[], w[][], q[][];
	
	int i, j, k;
	
	/*
	 * ALGORITMO GENEREC
	 * -----------------
	 * 
	 * sum1 = soma de todas as entradas (x[i]) multiplicadas com as sinpases entre entradas e camada escondida (w[i][j])
	 * sum2 = soma de todas as saidas desejadas (y[k]) multiplicadas com todas as sinapses entre camada escondida e saidas (q[j][k])
	 * sum3 = soma de todas as saidas calculadas (o[k]) multiplicadas com todas as sinapses entre camada escondida e saidas (q[j][k])
	 * sum4 = soma de todas as ativações "menos" (hMinus[j]) multiplicadas com todas as sinapses entre camada escondida e saidas (q[j][k])
	 * 
	 * hPlus = ativacoes na camada escondida da fase "mais" (sinal de treinamento)
	 * hMinus = ativacoes na camada escondida da fase "menos" (expectativa da rede)
	 * 
	 * N = taxa de aprendizado
	 */
	
	double sum1, sum2, sum3, sum4;
	
	double hPlus[], hMinus[];
	
	double N = 0.75;
	
	/*
	 * BACKPROPAGATION
	 * ---------------
	 * 
	 * u[k] = erros nas saidas (k = 0,..,C-1)
	 * s[j] = somas temporarias na camada escondida (j = 0,..,B-1)
	 * f[j] = erros na camada escondida (j = 0,..,B-1)
	 * 
	 * deltaQ[j][k] = ajustes calculados para sinapses entre camada escondida e saidas (j = 0,..,B-1; k = 0,..,C-1)
	 * deltaW[i][j] = ajustes calculados para sinapses entre camada de entrada e camada escondida (i = 0,..,A-1; j = 0,..,B-1)
	 *
	 * N = taxa de aprendizado
	 */
	
	double u[], s[], f[], deltaQ[][], deltaW[][];
	
	/* Arquivo de aprendizado */
	
	String fileName; 	// local do arquivo de aprendizado
	
	// modos de processamento
	
	final int VALIDATE = 1;
	final int TRAIN = 2;
	final int TEST = 3;
		
	int mode = TRAIN;	// modo atual e default
		
	/* Outras variaveis gerais */
	
	int ruido=0;	// para testes com ruido
	
	int algorithm = BACKPROPAGATION;	// definicao do algoritmo central - Backpropagation ou GeneRec
	
	String descIn[], descOut[];	// descricoes de entradas e saidas
	
	boolean errorFound = false;	// flag de erro de processamento 	
	
	/* para saida de console */
	
	ConsoleIO console;
	
	WinIO win;
	
	/**
	 * Construtor que inicializa a rede de acordo com os parametros fornecidos 
	 * @param numInputs numero de entradas
	 * @param numHidden numero de neuronios escondidos
	 * @param numOutputs numero de saidas
	 */
	
	public ANN(int numInputs, int numHidden, int numOutputs)
	{
		A=numInputs;
		B=numHidden;
		C=numOutputs;
		
		x = new double[A];
		h = new double[B];	s = new double[B]; f = new double[B];
		hPlus = new double[B]; hMinus = new double[B];
		o = new double[C];	
		y = new double[C];	u = new double[C];
		
		w = new double[A][B];	deltaW = new double[A][B];
		q = new double[B][C];	deltaQ = new double[B][C];
		
		/* inicializa as descricoes */
		
		descIn = new String[A];
		descOut = new String[C];
		
		/* inicializa a rede com valores randomicos nas sinapses (vide detalhes abaixo) */
		
		initNet();
		
		/* inicializa console */
		
		console = new ConsoleIO();
		
	}
	
	/**
	 * Mostrar titulo de trabalho em console
	 */
	
	public void titulo()
	{
		System.out.println("+-----------------------------------------+");
		System.out.println("| Sistema de Reconhecimento de Caracteres |");
		System.out.println("| Desenvolvido 2011 por Marvin Schneider  |");
		System.out.println("|      GNU General Public Licence         |");
		System.out.println("+-----------------------------------------+");
		System.out.println();
	}
	
	/**
	 * Zerar entradas e saidas desejadas (para presentar um novo par de entradas e saidas)
	 */
	
	public void zeroInputsDesiredOutputs()
	{
		for(int i=0;i<A;i++)
			x[i]=0;
		
		for(int k=0;k<C;k++)
			y[k]=0;	
	}
	
	/**
	 * Zerar ativacoes na camada escondida (para executar um passo de feedfoward)
	 */
	
	public void zeroHiddenOutputs()
	{
		for(int j=0;j<B;j++)
			h[j]=0;
		
		for(int k=0;k<C;k++)
			o[k]=0;
	}
	
	/**
	 * Primeira inicializacao das sinapses
	 */
	
	public void initSynapses()
	{
		/** inicializar as sinapses com valores entre -0.1 e 0.1 **/
		
		for(int i=0;i<A;i++)
			for(int j=0;j<B;j++)
				w[i][j]=(Math.random() * 10) / 200;
				
		for(int j=0;j<B;j++)
			for(int k=0;k<C;k++)
				q[j][k]=(Math.random() / 10) / 200;
		
	}
	
	/**
	 * "Inicializacao mestre" da rede inteira
	 */
	
	public void initNet()
	{
		zeroInputsDesiredOutputs();
		
		initSynapses();
		
		// Tambem zerar as descricoes
		
		for(int i=0;i<A;i++)	
			descIn[i]="";
		
		for(int k=0;k<C;k++)
			descOut[k]="";
	}
	
	/**
	 * 
	 */
	
	public void dumpEntradasSaidas()
	{
System.out.println("Entradas (x[i])");
		
		for(int i=0;i<A;i++)
			System.out.print("[" + i + "] = " + x[i] + "   ");
		
		System.out.println("\n");
		
		
		
		System.out.println("Camada escondida (h[j])");
		
		for(int j=0;j<B;j++)
			System.out.print("[" + j + "] = " + h[j] + "   ");
		
		System.out.println("\n");
		
		
		
		System.out.println("Saidas calculadas (o[k])");
		
		for(int k=0;k<C;k++)
			System.out.print("[" + k + "] = " + o[k] + "   ");
		
		System.out.println("\n");
		
		
		
		System.out.println("Saidas desejadas (y[k])");
		
		for(int k=0;k<C;k++)
			System.out.print("[" + k + "] = " + y[k] + "   ");
		
		System.out.println("\n");
	}
	
	/**
	 * Mostra os valores da rede atual em console
	 */
	
	public void dumpNet()
	{
		dumpEntradasSaidas();
				
		
		System.out.println("Sinapses entre entradas e camada escondida (w[i][j])");
		
		for(int i=0;i<A;i++)
		{
			for(int j=0;j<B;j++)
				System.out.print("[" + i + "/" + j + "] = " + w[i][j] + "   ");
			
			System.out.println();
		}
		
		System.out.println();
		
		
		
		System.out.println("Sinapses entre camada escondida e saidas (q[j][k])");
				
		for(int j=0;j<B;j++)
		{
			for(int k=0;k<C;k++)
				System.out.print("[" + j + "/" + k + "] = " + q[j][k] + "   ");
			
			System.out.println();
		}
		
		System.out.println();
		
		System.out.println("------------------------------------------------------\n");
	}
	
	/**
	 * Funcao de ativacao padrao (sigmoide)
	 * 
	 * @param x valor de entrada
	 * @return
	 */
	
	public double sigmoide(double x)
	{
		return (1 / (1 + Math.exp(-x)));
	}
	
	/**
	 * Executar um passo feedfoward
	 */
	
	public void feedForward()
	{
		// zerar ativacoes na camada escondida e saidas (entradas jah foram colocadas)
		
		zeroHiddenOutputs();
		
		// somas na camada escondida
		
		for(i=0;i<A;i++)
			for(j=0;j<B;j++)
				h[j]=h[j]+x[i]*w[i][j];
		
		// ativacao conforme somas
		
		for(j=0;j<B;j++)
			h[j]=sigmoide(h[j]);
		
		// somas na camada de saida
		
		for(j=0;j<B;j++)
			for(k=0;k<C;k++)
				o[k]=o[k]+h[j]*q[j][k];
		
		// ativacoes conforme somas
		
		for(k=0;k<C;k++)
			o[k]=sigmoide(o[k]);
		
	}
	
	/**
	 * Executa algoritmo BackPropagation
	 */
	
	public void backPropagation()
	{
		// (Saidas desejadas e entradas jah foram fornecidas ah rede; a expectativa da rede foi calculada) 
		
		/* Definir erros nas saidas */
		
		for(k=0;k<C;k++)
			u[k]=o[k]*(1.0-o[k])*(y[k]-o[k]);
		
		/* Definir erros na camada escondida */
		
		for(j=0;j<B;j++)
		{
			s[j]=0.0;
			
			for(k=0;k<C;k++)
				s[j]=s[j]+u[k]*q[j][k];
			
			f[j]=h[j]*(1.0-h[j])*s[j];
		}
		
		/* Diferenca nas sinapses entre camada escondida e saidas */
		
		for(j=0;j<B;j++)
			for(k=0;k<C;k++)
				deltaQ[j][k]=N*u[k]*h[j];
		
		/* Diferenca nas sinapses entre entradas e camada escondida */
		
		for(i=0;i<A;i++)
			for(j=0;j<B;j++)
				deltaW[i][j]=N*f[j]*x[i];
		
		/* Fazer os ajustes */
		
		for(j=0;j<B;j++)
			for(k=0;k<C;k++)
				q[j][k]=q[j][k]+deltaQ[j][k];
		
		for(i=0;i<A;i++)
			for(j=0;j<B;j++)
				w[i][j]=w[i][j]+deltaW[i][j];
		
	}
	
	/**
	 * Escrever resultados do algoritmo BackPropagation em console
	 */
	
	public void dumpBackPropagation()
	{
		System.out.println("Erros nas saidas (u[k])");
		
		for(int k=0;k<C;k++)
			System.out.print("[" + k + "] = " + u[k] + "   ");
		
		System.out.println("\n");
		
		
		System.out.println("Somas temporarias na camada escondida (s[j])");
		
		for(int j=0;j<B;j++)
			System.out.print("[" + j + "] = " + s[j] + "   ");
		
		System.out.println("\n");
		
		
		System.out.println("Erros na camada escondida (f[j])");
		
		for(int j=0;j<B;j++)
			System.out.print("[" + j + "] = " + f[j] + "   ");
		
		System.out.println("\n");
		
		
		
		System.out.println("Diferenca nas sinapses entre camada escondida e saidas (deltaQ[j][k])");
				
		for(int j=0;j<B;j++)
		{
			for(int k=0;k<C;k++)
				System.out.print("[" + j + "/" + k + "] = " + deltaQ[j][k] + "   ");
			
			System.out.println();
		}
		
		System.out.println();
		
		
		
		System.out.println("Diferenca nas sinapses entre camada de entrada e camada escondida (deltaW[i][j])");
		
		for(int i=0;i<A;i++)
		{
			for(int j=0;j<B;j++)
				System.out.print("[" + i + "/" + j + "] = " + deltaW[i][j] + "   ");
			
			System.out.println();
		}
		
		System.out.println();
		
	
	}
	
	/**
	 * Executar GeneRec
	 * 
	 * Se "onlyMinus" for verdadeiro, apenas a expectativa da rede é calculada
	 * 
	 */
	
	public void geneRec(boolean onlyMinus)
	{
		
		sum1=0; sum2=0; sum3=0; sum4=0;
		
		for(int j=0;j<B;j++)
		{
			for(int i=0;i<A;i++)
			{
				sum1=sum1+w[i][j]*x[i];
			}
			
			for(int k=0;k<C;k++)
			{
				sum2=sum2+q[j][k]*y[k];
			}
			
			for(int k=0;k<C;k++)
			{
				sum3=sum3+q[j][k]*o[k];
			}
			
			if(onlyMinus==false) hPlus[j]=sigmoide(sum1+sum2);	// sinal de ativacao positivo (1)
			
			hMinus[j]=sigmoide(sum1+sum3); // sinal de ativacao negativo (2)
			
			sum1=0;
			sum2=0;
			sum3=0;
		}
		
		sum4=0;
		
		for(int k=0;k<C;k++)
		{
			for(int j=0;j<B;j++)
			{
				sum4=sum4+q[j][k]*hMinus[j];	// proxima ativacao (3)
			}
			
			o[k]=sigmoide(sum4);
			
			sum4=0;
			
		}
		
		if(onlyMinus==false)
		{
		
			for(int j=0;j<B;j++)
			{
				for(int k=0;k<C;k++)
				{
					q[j][k]=q[j][k]+N*(y[k]-o[k])*hMinus[j];	// ajuste 1 (4)
				}
			}
		
			for(int i=0;i<A;i++)
			{
				for(int j=0;j<B;j++)
				{
					w[i][j]=w[i][j]+N*(hPlus[j]-hMinus[j])*x[i];	// ajuste 2 (5)
				}
			}
		}
	}
	
	/**
	 * Colocar um valor de entrada em "1"
	 * 	  
	 * @param pos posicao da entrada
	 */
	
	public void setInput(int pos, int value)
	{
		if(pos<A && pos>-1)
			
			x[pos]=value;
		
		else
			
			System.out.println("\n>>Error: Impossivel colocar entrada = 1 na posicao = " + pos + "(fora de intervalo).\n");
	}
	
	/**
	 * Retorna um valor de saida calculada de maneira individual
	 * 
	 * @param i posicao do valor a retornar
	 * @return
	 */
		
	public double getOutput(int i)
	{
		return o[i];
	}
	
	/**
	 * Coloca um valor individual na saida "i" 
	 * @param i posicao
	 * @param d valor
	 */
	
	public void setOutput(int i, double d)
	{
		o[i]=d;
	}
	
	/**
	 * Coloca "1" numa posicao de saida desejada
	 * 
	 * @param pos posicao
	 */
	
	public void setSaidaDesejada(int pos)
	{
		if(pos<C && pos>-1)
			
			y[pos]=1;
		
		else
			
			System.out.println("\n>>Erro: Impossivel colocar '1' em saida desejada na posicao " + pos + "(fora de intervalo).\n");
	}
	
	/**
	 * Retorna valor de sinapse (w[i][j])
	 * @param i i
	 * @param j j
	 * @return
	 */
	
	public double getW(int i, int j)
	{
		return w[i][j];
	}
	
	/**
	 * Define valor de sinapse (w[i][j])
	 * @param i i 
	 * @param j j 
	 * @param value valor
	 */
	
	public void setW(int i, int j, double value)
	{
		w[i][j]=value;
	}
	
	/**
	 * Retorna valor de sinapse (q[j][k])
	 * @param j j
	 * @param k k
	 * @return
	 */
	
	public double getQ(int j, int k)
	{
		return q[j][k];
	}
	
	/**
	 * Define valor de sinapse (q[j][k])
	 * @param j j
	 * @param k k
	 * @param value valor
	 */
	
	public void setQ(int j, int k, double value)
	{
		q[j][k]=value;
	}
	
	/**
	 * Mensagem de erro de sintaxe na leitura de arquivo
	 *  
	 * @param message esperado
	 * @param message2 lido
	 */
	
	public void syntaxError(String message, String message2)
	{
		System.out.println("[!ATENCAO!] Erro de sintaxe no arquivo! Esperado " + message + ".\n");
		System.out.println("[!ATTENTION!] Lido: " + message2 + "\n");
		System.out.println("[!ATTENTION!] Interrompendo processo de leitura de arquivo.....\n");
		
		errorFound = true;
	}
	
	/**
	 * Escreve uma mensagem em console
	 * 
	 * @param message mensagem a ser escrita
	 */
	
	public void msg(String message)
	{
		if(mode==VALIDATE) 
			System.out.println("[.MENSAGEM.] " + message + ".\n");
	}
	
	/**
	 * Conta o numero de valores de uma linha e checa a formatacao
	 * 
	 * @param values linha lida
	 * @return
	 */
	
	public int formatOk(String values)
	{
		int number = 0;
		
		for(int g=0;g<values.length();g++)
		{
			// percorrer os valores da linha
			
			if(values.charAt(g)=='0' || values.charAt(g)=='1' || values.charAt(g)==',')	// apenas esses sao validos
			{
				if(g%2==0)	// 0 ou 1 esperado
				{
					if(values.charAt(g)==',')
						return -1;	// virgual interrompe
					
					number++;	// mais um valor contado
					
				} else
					if(values.charAt(g)!=',')	// virgula esperada  	
						return -1;
			} else
				if(g==values.length())
					if(values.charAt(g)!=';')	// no final, ponto e virgula
						return -1;
		}
		
		return number;	// retorna o numero de valores achados de acordo com o formato
	}
	
	/**
	 * Transfere valores de arquivo para entrada da rede
	 * 
	 * @param inStr string de valores a transferir
	 */
	
	public void valuesToInputs(String inStr)
	{
		// Nesse ponto tudo deve estar checado, entao nao haverah erro a testar na hora
		
		int valor = 0;
		
		for(int g=0;g<inStr.length();g++)
		{
			if(g%2 == 0)
			{
				if(inStr.charAt(g)=='1' || inStr.charAt(g)=='0')
				{
				
					if(inStr.charAt(g)=='1') x[valor]=1;
					
					if(inStr.charAt(g)=='0') x[valor]=0;
					
					valor++;
				}
			}
		}
	}
	
	/**
	 * Transfers values from file to output of network
	 * 
	 * @param inStr string of values to be transferred
	 */
	
	public void valuesToOutputs(String inStr)
	{
		// neste ponto jah deve haver tudo checado, entao sem problemas com "traducao direta"
		
		int valor = 0;
		
		for(int g=0;g<inStr.length();g++)
		{
			if(g%2 == 0)
			{
				if(inStr.charAt(g)=='1' || inStr.charAt(g)=='0')
				{
				
					if(inStr.charAt(g)=='1') y[valor]=1;
					
					if(inStr.charAt(g)=='0') y[valor]=0;
					
					valor++;
				}
			}
		}
	}
	
	/**
	 * Rotina de teste que causa ruido na entrada
	 * 
	 */
	
	public void addNoise()
	{
		int pos = 0;
		
		for(int b=0;b<ruido;b++)
		{
			// pegar posicao randomica de acordo com o numero de elementos ruido definidos acima
			
			pos=(int)(Math.random()*A);
			
			if(x[pos]==0) x[pos]=1; else x[pos]=0;
		}
		
		System.out.println(pos + " ");	// informando a posicao (apenas debug)
	}
	
	/**
	 * Rotina que leh o arquivo de treinamento e transfere os dados para entradas e saidas da rede
	 * 
	 * @param nameOfFile name of file to be read
	 */
	
	public int trainFile(String nameOfFile)
	{
		/**
		 * FORMATO DE ARQUIVO ESPERADO (EXEMPLO)
		 * -------------------------------------
		 * 
		 * [PAIR 1]
		 * *In*
		 * 1,0,0,0,0,1,1,1,0,1;
		 * *Out*
		 * 0,0,0,1,1,0,1,0,1,0;
		 *
		 * [PAIR2]
		 * *In*
		 * 1,1,0,1,0,1,1,1,0,1;
		 * *Out*
		 * 0,0,0,0,0,0,0,1,1,1;
		 *
		 * .....
		 */
		
		// interno para leitura de arquivo
		
		FileReader fr = null;
		BufferedReader br;
		String str, pair = null;
		
		int setsRead = 0;	// conjuntos lidos
		double totalError = 0.0;	// erro total
		int wrongAnswers = 0;	// respostas erradas
		
		// estado da leitura (qual entrada é esperada como proxima)
				
		final int EXPECTING_PAIR = 0;	// esperando [PAIR..]
		final int EXPECTING_INPUT = 1;  // esperando *In*
		final int EXPECTING_INPUT_VALUES = 2;	// esperando valores de entrada
		final int EXPECTING_OUTPUT = 3;	// esperando *Out*
		final int EXPECTING_OUTPUT_VALUES = 4;	// esperando valores de saida
		
		int state = EXPECTING_PAIR;	// no inicio sempre espera-se [PAIR..]
		
		int returnedValues = 0;
		
		// Abertura do arquivo
		
		try {
			fr = new FileReader(nameOfFile);
		} catch (FileNotFoundException e) {
			
			console.showError("Impossivel abrir o arquivo " + nameOfFile);
			
		}
		
		br = new BufferedReader(fr);
		
		errorFound = false;
		
		// Leitura de linhas ateh o final do arquivo
		
		try {
			while((str = br.readLine()) !=null && errorFound == false)
			{
				//System.out.println(str);
				
				switch(state)
				{
				case EXPECTING_PAIR:
					
					if(str.equals("")) break; // eliminar espacos em branco antes de [PAIR
					
					if(str.startsWith("[PAIR")){
					
						pair = str.substring(4, str.length()-1);	// valor apos [PAIR
						
						msg("Lendo par " + pair);
						
						state = EXPECTING_INPUT;
						
						break;
					}
					
					syntaxError("[PAIRn]",str);
					
					break;
					
				case EXPECTING_INPUT:
					
					if(str.equals(""))	// Eliminiar espaços antes de *In*
						break;
					
					if(str.equals("*In*"))
					{
						msg("Input encontrado");
						
						state = EXPECTING_INPUT_VALUES;
						
						break;
					}
					
					syntaxError("*In*",str);
					
					break;
					
				case EXPECTING_INPUT_VALUES:
					
					if(str.equals(""))	// blanks may be inserted before the input values
						break;
				
					returnedValues = formatOk(str);
					
					if(returnedValues>0)
					{
						msg("Formato ok, valores de entrada encontrados: " + returnedValues);
						
						if(returnedValues==A)
						{
							msg("Numero de valores encontrados nao confere com o esperado.");
						} else
							syntaxError(A + " input values.", str);
						
						if(mode==TRAIN)		// treinando, entao valores nas entradas!
						{
							valuesToInputs(str);
						}
						
						if(mode==TEST)		// testando, entao valores nas entradas!
						{
							valuesToInputs(str);
						}
					
						state = EXPECTING_OUTPUT;
					
						break;
					}
					
					syntaxError("Formato de valores.",str);
					
				case EXPECTING_OUTPUT: 
					
					if(str.equals(""))	// eliminar espacos antes de *Out*
						break;
					
					if(str.equals("*Out*"))
					{
						msg("Output encontrado");
						
						state = EXPECTING_OUTPUT_VALUES;
						
						break;
					}
					
					syntaxError("*Out*",str);
					
					break;
					
				case EXPECTING_OUTPUT_VALUES:
				
					if(str.equals(""))	// blanks may be inserted before output values
						break;
				
					returnedValues = formatOk(str);
					
					if(returnedValues>0)
					{
						msg("Formato ok, valores de saida encontrados: " + returnedValues);
						
						if(returnedValues==C)
						{
							msg("Numero de valores nao confere com o esperado.");
						} else
							syntaxError(C + " output values", str);
					
						if(mode==TRAIN)	// treinar, entao valores para saidas!
						{
							valuesToOutputs(str);
							
							if(algorithm==BACKPROPAGATION)	// chaveamento do algoritmo usado
							{
							
								feedForward();		// Backpropagation
							
								backPropagation();
							} else
							{
								geneRec(false);		// GeneRec
							}
						}
						
						if(mode==TEST)	// modo teste, entao valores para saidas!
						{
							setsRead++;
							
							valuesToOutputs(str);
							
							if(ruido>0)		// querendo ruido, ele eh inserido aqui
							{
								addNoise();
							}
							
							if(algorithm==BACKPROPAGATION) 	// chaveamento de algoritmo
								feedForward(); else
									geneRec(true);
							
							for(int t=0;t<C;t++)	// calculo de erro
							{
								if(y[t]==1 && o[t]<0.5) wrongAnswers++;		// resposta errada (isto eh, escolha errada)
								if(y[t]==0 && o[t]>0.5) wrongAnswers++;
								
								totalError=totalError+Math.abs(y[t]-o[t]);	// erro absoluto
								
								//dumpEntradasSaidas();
								
								//console.readString();
							}
						}
						
						state = EXPECTING_PAIR;	// proximo par!
					
						break;
					}
					
					syntaxError("Formato de valores.",str);
				
				}
			}
		} catch (IOException e) {
			
			console.showError("Impossivel ler arquivo. " + nameOfFile);
			
		}
		
		// Fechamento do arquivo
		
		try {
			br.close();
		} catch (IOException e) {
			
			console.showError("Arquivo nao pode ser fechado. " + nameOfFile);
			
		}
		
		if(mode==TEST)
		{
			System.out.println("RESULTADOS DO TESTE");
			System.out.println("-------------------\n");
			
			System.out.println("Conjuntos lidos:      " + setsRead);
			System.out.println("Erro total abs(x[]-o[]):    " + totalError);
			System.out.println("Erro relativo (erro total por cada saida): " + (totalError/(setsRead*C)));
			System.out.println("Escolhas erradas:  " + wrongAnswers + "\n\n");	
			
			return wrongAnswers;
		}
		
		return -1;
	}
	
	/**
	 * Permite ao usuario alterar as descricoes das entradas
	 * 
	 */
	
	public void readDescIn()
	{
		for(int i=0;i<A;i++)
		{
			System.out.print("Input Description " + i + " : ");
			descIn[i]=console.readString();
		}
	}
	
	/**
	 * Permite ao usuario alterar as descricoes das saidas
	 * 
	 */
	
	public void readDescOut()
	{
		for(int k=0;k<C;k++)
		{
			System.out.print("Output Description " + k + " : ");
			descOut[k]=console.readString();
		}
	}
	
	/**
	 * Define o nome do arquivo a ser lido durante o treinamento
	 * 
	 */
	
	public void readFileName()
	{
		System.out.print("Nome do arquivo (com caminho completo!) : ");
		fileName=console.readString();
	}
	
	public void defineAlgoritmo(int algoritmo)
	{
		algorithm = algoritmo;
	}
	
	public void defineTaxaAprendizado(double n)
	{
		this.N=n;
	}
	
	class RecListener implements ActionListener {

		JButton button;
		
		public RecListener(JButton button)
		{
			this.button = button;
		}
		@Override
		public void actionPerformed(ActionEvent arg0) {
			// TODO Auto-generated method stub
			if(button.getBackground()==Color.BLACK)
				button.setBackground(Color.WHITE); else
					button.setBackground(Color.BLACK);
		}
		
	}
	
	class FecharListener implements ActionListener
	{

		@Override
		public void actionPerformed(ActionEvent e) {
			System.exit(0);
			
		}
		
	}
	
	/* Execucao do Reconhecimento */
	
	class ReconhecerListener implements ActionListener
	{
		JButton[][] recButtonLocal;
		
		public ReconhecerListener(JButton[][] recButtonLocal)
		{
			this.recButtonLocal = recButtonLocal;
		}
		
		/* Colocar Dados da Tela nas Entradas da Rede */
		
		public void traduzirEntrada()
		{
			int entradaNum = 0;
			
			zeroInputsDesiredOutputs();
			
			for(int j=0;j<10;j++)
				for(int i=0;i<10;i++)
				{
					if(recButtonLocal[i][j].getBackground()==Color.BLACK)
						x[entradaNum]=1; else x[entradaNum]=0;
					entradaNum++;
				}
			
			//dumpEntradasSaidas();
		}
		
		/* Executar Passo de Reconhecimento da Rede */
		
		public void funcionarRede()
		{
			zeroHiddenOutputs();
			
			if(algorithm==BACKPROPAGATION) 	// chaveamento de algoritmo
				feedForward(); else
					geneRec(true);
		}
		
		/* Informar o Valor da Saida (a Letra Reconhecida) */
		
		public void decodificaSaida()
		{
			WinIO win = new WinIO();
			
			String saidaStr = ""; 
			
			for(int l=0;l<5;l++)
			{
				if(o[l]<0.5)
					saidaStr = saidaStr + "0"; else
						saidaStr = saidaStr + "1";
			}
			
			/* Decodificacao de Binarios */
			
			if(saidaStr.equals("00000")) win.message("Letra reconhecida", "A"); else
			if(saidaStr.equals("00001")) win.message("Letra reconhecida", "B"); else
			if(saidaStr.equals("00010")) win.message("Letra reconhecida", "C"); else
			if(saidaStr.equals("00011")) win.message("Letra reconhecida", "D"); else
			if(saidaStr.equals("00100")) win.message("Letra reconhecida", "E"); else
			if(saidaStr.equals("00101")) win.message("Letra reconhecida", "F"); else
			if(saidaStr.equals("00110")) win.message("Letra reconhecida", "G"); else
			if(saidaStr.equals("00111")) win.message("Letra reconhecida", "H"); else
			if(saidaStr.equals("01000")) win.message("Letra reconhecida", "I"); else
			if(saidaStr.equals("01001")) win.message("Letra reconhecida", "J"); else
			if(saidaStr.equals("01010")) win.message("Letra reconhecida", "K"); else
			if(saidaStr.equals("01011")) win.message("Letra reconhecida", "L"); else
			if(saidaStr.equals("01100")) win.message("Letra reconhecida", "M"); else
			if(saidaStr.equals("01101")) win.message("Letra reconhecida", "N"); else
			if(saidaStr.equals("01110")) win.message("Letra reconhecida", "O"); else
			if(saidaStr.equals("01111")) win.message("Letra reconhecida", "P"); else
			if(saidaStr.equals("10000")) win.message("Letra reconhecida", "Q"); else
			if(saidaStr.equals("10001")) win.message("Letra reconhecida", "R"); else
			if(saidaStr.equals("10010")) win.message("Letra reconhecida", "S"); else
			if(saidaStr.equals("10011")) win.message("Letra reconhecida", "T"); else
			if(saidaStr.equals("10100")) win.message("Letra reconhecida", "U"); else
			if(saidaStr.equals("10101")) win.message("Letra reconhecida", "V"); else
			if(saidaStr.equals("10110")) win.message("Letra reconhecida", "W"); else
			if(saidaStr.equals("10111")) win.message("Letra reconhecida", "X"); else
			if(saidaStr.equals("11000")) win.message("Letra reconhecida", "Y"); else
			if(saidaStr.equals("11001")) win.message("Letra reconhecida", "Z"); else
					win.message("Letra reconhecida", "Nenhuma indicacao valida");
		}

		@Override
		public void actionPerformed(ActionEvent e) {
			traduzirEntrada();
			funcionarRede();
			decodificaSaida();
		}
		
	}
	
	/* Limpar a Tela de Reconhecimento */
	
	class LimparListener implements ActionListener
	{
		JButton[][] recButtonLocal;
		
		public LimparListener(JButton[][] recButtonLocal)
		{
			this.recButtonLocal=recButtonLocal;
		}

		@Override
		public void actionPerformed(ActionEvent e) {
			// TODO Auto-generated method stub
			for(int i=0;i<10;i++)
				for(int j=0;j<10;j++)
					recButtonLocal[i][j].setBackground(Color.WHITE);
		}
	}
	
	/* Rotina Mestre para Reconhecimento de Caracteres */
	
	public void reconhecimento()
	{
		/* Montagem da Tela */
		
		Window reconWin = new Window("Reconhecimento");
		
		JButton[][] recButton;
		
		JPanel panel = new JPanel();
		
		JButton reconhecer  = new JButton("Reconhecer");
		JButton fechar = new JButton("Fechar");
		JButton limpar = new JButton("Limpar");
		
		fechar.addActionListener(new FecharListener());
				
		panel.add(reconhecer);
		panel.add(limpar);
		panel.add(fechar);
			
		recButton = new JButton[10][10];
		
		/* Adicionar os Botoes e suas Funcionalidades */
		
		for(int i=0;i<10;i++)
			for(int j=0;j<10;j++)
			{
				recButton[i][j] = new JButton("  ");
				recButton[i][j].setBackground(Color.WHITE);
				recButton[i][j].addActionListener(new RecListener(recButton[i][j]));
				reconWin.add(recButton[i][j], i, j, 1, 1);
			}
		
		limpar.addActionListener(new LimparListener(recButton));
		reconhecer.addActionListener(new ReconhecerListener(recButton));
		
		reconWin.add(panel, 0, 10, 10, 1);
		
		reconWin.setSize(400, 320);
		
		reconWin.centerPos();
		
		reconWin.show();	/* Mostrar Janela */
	}
	
	/* programa principal */
	
	public static void main(String[] args)
	{
		ANN ann = new ANN(100,10,5);	/* criacao da rede */
		
		WinIO win = new WinIO();
		
		String fileName =  "";
		
		JFileChooser fc = new JFileChooser();
		
		int command = fc.showOpenDialog(null);
		
		if(command==JFileChooser.APPROVE_OPTION)
		{
			fileName = fc.getSelectedFile().getAbsolutePath();
			
			if(fc.getSelectedFile().exists()==false)
				System.exit(0);
		}
		
		// Definicao de iteracoes de aprendizado
		
		int iteracoes = win.readNumber("Numero de Iteracoes");
		
		// Definicao da taxa de aprendizado
		
		double n = win.readDouble("Taxa de Aprendizado (0.01 - 0.99)");
		
		ann.defineTaxaAprendizado(n);
		
		// Escolha do algoritmo
		
		Object options[] = {"Backpropagation","GeneRec"};
		
		int algoritmo = win.openQuestion("Usar qual algoritmo?", "Algoritmo", options);
		
		if(algoritmo==0) 
			ann.defineAlgoritmo(ann.BACKPROPAGATION); else
				ann.defineAlgoritmo(ann.GENEREC);
		
		// Treinamento da rede conforme configuracao acima
		
		ann.mode=ann.TRAIN;
		
		for(int i=0;i<iteracoes;i++)
		{
				ann.trainFile(fileName);
		}
		
		// Resultados do treinamento em console
		
		ann.mode=ann.TEST;
		ann.trainFile(fileName);
		
		ann.reconhecimento();
		
	}
	
}
