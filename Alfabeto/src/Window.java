
/*
 * Class created on 27/04/2005
 *
 */
import javax.swing.*;

import java.awt.*;
/**
 * Utilitario para facilitar o use do GridBagLayout dentro do Java
 * 
 * written by Marvin Oliver Schneider
 * 
 * Touched again in 2011 with several improvements (V3)
 *
 */
public class Window {
	
	JFrame frame;				// frame a ser criado
	Container frameContainer;	// container para o frame
	
	public final int HORIZONTAL = GridBagConstraints.HORIZONTAL;	// direcoes no layout
	public final int NONE       = GridBagConstraints.NONE;
	public final int CENTER     = GridBagConstraints.CENTER;
	public final int WEST       = GridBagConstraints.WEST;
	public final int EAST       = GridBagConstraints.EAST;
	public final int BOTH       = GridBagConstraints.BOTH;
	
	public final int MARGIN_X = 20;
	public final int MARGIN_Y = 80;
	
	public static final int MIDDLE = 0;
	public static final int BELOW = 1;
	public static final int ABOVE = 2;
	public static final int LEFT = 3;
	public static final int RIGHT = 4;
	
	int int_x = 0, int_y = 0;
	
	/**
	 * O contrutor cria o frame colocando um texto no topo dele e vincula a ele o GridBagLayout
	 */
	
	public Window(String texto){
		
		frame = new JFrame(texto);
		frameContainer = frame.getContentPane();
		frameContainer.setLayout(new GridBagLayout());
		
	}
	
	/**
	 * Adicionar um elemento ao frame com controle total dos parametros
	 */
	
	public void addExtended(Component comp, int x, int y, int width, int height, int fill, int anchor){
		Elemento.add(frameContainer,comp,x,y,width,height,1,1,fill,anchor);
	}
	
	/**
	 * Adicionar um elemento ao frame da maneira padrao
	 */
	
	public void add(Component comp, int x, int y, int width, int height){
		Elemento.add(frameContainer,comp,x,y,width,height,1,1,NONE,CENTER);
	}
	
	public void addLeft(Component comp) {
		Elemento.add(frameContainer, comp, int_x, int_y, 1, 1, 1, 1, NONE, WEST);
		int_x++;
	}
	
	public void addLeftLn(Component comp) {
		Elemento.add(frameContainer, comp, int_x, int_y, 1, 1, 1, 1, NONE, WEST);
		int_x++;
		cr();
	}
	
	public void addRight(Component comp) {
		Elemento.add(frameContainer, comp, int_x, int_y, 1, 1, 1, 1, NONE, EAST);
		int_x++;
	}
	
	public void addRightLn(Component comp) {
		Elemento.add(frameContainer, comp, int_x, int_y, 1, 1, 1, 1, NONE, EAST);
		int_x++;
		cr();
	}
	
	public void addCenter(Component comp) {
		Elemento.add(frameContainer, comp, int_x, int_y, 1, 1, 1, 1, NONE, CENTER);
		int_x++;
	}
	
	public void addCenterLn(Component comp) {
		Elemento.add(frameContainer, comp, int_x, int_y, 1, 1, 1, 1, NONE, CENTER);
		int_x++;
		cr();
	}
	
	public void addRangeLeft(Component comp, int x, int length)
	{
		Elemento.add(frameContainer, comp, x, int_y, length, 1, 1, 1, NONE, WEST);
		int_x = x + length;
	}
	
	public void addRangeLeftLn(Component comp, int x, int length)
	{
		Elemento.add(frameContainer, comp, x, int_y, length, 1, 1, 1, NONE, WEST);
		int_x = x + length;
		cr();
	}
	
	public void addRangeRight(Component comp, int x, int length)
	{
		Elemento.add(frameContainer, comp, x, int_y, length, 1, 1, 1, NONE, EAST);
		int_x = x + length;
	}
	
	public void addRangeRightLn(Component comp, int x, int length)
	{
		Elemento.add(frameContainer, comp, x, int_y, length, 1, 1, 1, NONE, EAST);
		int_x = x + length;
		cr();
	}
	
	public void addRangeCenter(Component comp, int x, int length)
	{
		Elemento.add(frameContainer, comp, x, int_y, length, 1, 1, 1, NONE, CENTER);
		int_x = x + length;
	}
	
	public void addRangeCenterLn(Component comp, int x, int length)
	{
		Elemento.add(frameContainer, comp, x, int_y, length, 1, 1, 1, NONE, CENTER);
		int_x = x + length;
		cr();
	}
	
	public void addLineLn(int length) {
		int_y++;
		Elemento.add(frameContainer, new JSeparator(), 0, int_y, length, 1, 1, 1, BOTH, CENTER);
		int_y++;
	}
	
	public void addLineNoExtendLn(int length) {
		int_y++;
		Elemento.add(frameContainer, new JSeparator(), 0, int_y, length, 1, 1, 1, NONE, CENTER);
		int_y++;
	}
	
	public void addInput4Ln(String text1, 
			Component comp1, 
			String text2, 
			Component comp2)
	{
		text1 = text1 + " ";
		text2 = text2 + " ";
		addRight(new JLabel(text1));
		addLeft(comp1);
		addRight(new JLabel(text2));
		addLeft(comp2);
		cr();
	}
	
	public void addInput2Ln(String text, Component comp)
	{
		text = text + " ";
		addRight(new JLabel(text));
		addLeft(comp);
		cr();
	}
	
	public void addSectionTitleLn(String text, int length)
	{
		addLineLn(length);
		
		addRangeCenterLn(new JLabel(text), 0, length);
		
		addLineLn(length);
	}
	
	public void addTitle(String text, int length)
	{
		addRangeCenterLn(new JLabel(text), 0, length);
		
		addLineLn(length);
	}
	
	public void pack()
	{
		frame.pack();
	}
	
	public void cr()
	{
		int_x=0;
		int_y++;
	}
	
	public void setPos(int x, int y)
	{
		int_x = x;
		int_y = y;
	}
	
	public void setPosWindow(int x, int y)
	{
		frame.setLocation(x, y);
	}
	
	/**
	 * Mostrar o frame na tela
	 */
	
	public void show(){
		frame.setVisible(true);
		
		frame.setAlwaysOnTop(true);
		frame.setAlwaysOnTop(false);
		
		setResizable(false);
	}
	
	/**
	 * Tirar o frame da tela
	 */
	
	public void hide(){
		frame.setVisible(false);
	}
	
	/**
	 * Configurar o tamanho do frame
	 */
	
	public void setSize(int x, int y){
		frame.setSize(x,y);
	}
	
	public int getY()
	{
		return frame.getSize().height;
	}
	
	public int getX()
	{
		return frame.getSize().width;
	}
	
	public void doubleSizePack()
	{
		pack();
		
		setSize(getX()*2,getY()*2);
	}
	
	public void standardPack()
	{
		int x, y;
		
		pack();
		
		x = (int) (getX() * 1.2);
		y = (int) (getY() * 1.5);
		
		setSize(x,y);
	}
	
	public void standardPackSmall()
	{
		int x, y;
		
		pack();
		
		x = (int) (getX() * 1.1);
		y = (int) (getY() * 1.3);
		
		setSize(x,y);
	}
	
	public void setResizable(boolean enable)
	{
		frame.setResizable(enable);
	}
	
	public void centerPos()
	{
		frame.setLocationRelativeTo(null);
	}
	
	public JFrame getFrame()
	{
		return frame;
	}
	
	public void setRelativePos(int position)
	{
		WinIO IO = new WinIO();
		
		int xFinal=0, yFinal=0;
		
		switch(position)
		{
		case MIDDLE: xFinal=(IO.getScreenWidth()-frame.getWidth())/2; 
						yFinal=(IO.getScreenHeight()-frame.getHeight())/2; 
						break;
		case BELOW: xFinal=(IO.getScreenWidth()-frame.getWidth())/2; 
					yFinal=(IO.getScreenHeight()-MARGIN_Y-frame.getHeight());
			break;
		case ABOVE: xFinal=(IO.getScreenWidth()-frame.getWidth())/2;
					yFinal=MARGIN_Y;
			break;
		case LEFT: xFinal=MARGIN_X; 
					yFinal=(IO.getScreenHeight()-frame.getHeight())/2;
					break;
		case RIGHT: xFinal=(IO.getScreenWidth()-MARGIN_X-frame.getWidth());
					yFinal=(IO.getScreenHeight()-frame.getHeight())/2;
					break;
		}
		
		setPosWindow(xFinal, yFinal);
	}
	
	/**
	 * Class that defines the element
	 */
	
	static class Elemento {
		static GridBagConstraints cons = new GridBagConstraints();
		
		public static void add(Container cont, Component comp, int x, int y, int width, int height, int weightx, int weighty, int fill, int anchor){
			cons.gridx=x;
			cons.gridy=y;
			cons.gridwidth=width;
			cons.gridheight=height;
			cons.weightx=weightx;
			cons.weighty=weighty;
			cons.fill=fill;
			cons.anchor=anchor;
			cont.add(comp, cons);	
		}
	}
}


