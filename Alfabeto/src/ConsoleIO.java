
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Random;

/**
 * Classe para gestão de console
 *
 */

public class ConsoleIO {
	
	BufferedReader d;
	
	Random randomGen;
	
	public ConsoleIO()
	{
		d = new BufferedReader(new InputStreamReader(System.in));
		
		randomGen = new Random();
	}
	
	/**
	 * Produz um valor randomico
	 * @param range intervalo (0-range)
	 * @return
	 */
	
	public int getRandom(int range)
	{
		return randomGen.nextInt(range);
	}
	
	/**
	 * Mostra uma mensagem de erro na tela
	 * 
	 * @param message variable message to be displayed
	 */
	
	public void showError(String message)
	{
		System.out.println("[!ATENCAO!] Erro! " + message);
	}
	
	/**
	 * Leh um numero inteiro positivo do teclado 
	 * @return
	 */
	
	public int readNumber()
	{
		String resposta = "";
		
		int flag;
		
		try {
			resposta=d.readLine();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			showError("Impossivel ler do teclado.");
		}
		
		try{
		
		flag = Integer.parseInt(resposta);
		
		} catch(Exception e) {
			
			showError("Por favor forneca um numer inteiro positivo.");
			
			flag = -999;
		}
		
		return flag;
		
	}
	
	/**
	 * Leh um valor real do teclado
	 * 
	 * @return
	 */
	
	public double readDouble()
	{
		String resposta = "";
		
		double flag;
		
		try {
			resposta=d.readLine();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			showError("Impossivel ler do teclado.");
		}
		
		try{
		
		flag = Double.parseDouble(resposta);
		
		} catch(Exception e) {
			
			showError("Por favor forneca um numero real.");
			
			flag = -999;
		}
		
		return flag;
		
	}
	
	/**
	 * Leh uma string do teclado
	 * 
	 * @return
	 */
	
	public String readString()
	{
		String resposta = "";
		
		try {
			resposta=d.readLine();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			showError("Impossivel ler do teclado.");
		}
		
		return resposta;
		
	}
	
}
