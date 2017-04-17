\documentclass[11pt, oneside]{article}   	
\usepackage[parfill]{parskip}    		% Activate to begin paragraphs with an empty line rather than an indent
\usepackage{graphicx}
\usepackage{float}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[dutch]{babel}
\usepackage{titling}
\newcommand{\subtitle}[1]{%
	\posttitle{%
		\par\end{center}
	\begin{center}\large#1\end{center}
	\vskip0.5em}%
}
\usepackage{tikz}
\usetikzlibrary{arrows}
\usepackage{titling}
\setlength{\droptitle}{-10em} 

\title{Expressiebomen}
\subtitle{Programmeren in de Wiskunde}
\author{Nina Bezem, Arja Blok en Isabel Droste}
\date{14 april 2017}
\begin{document}
\maketitle
\thispagestyle{empty}

\begin{abstract}
	\noindent Het rekenen met expressies is belangrijk voor het uitvoeren van ingewikkelde berekeningen. Hiervoor kan gebruik gemaakt worden van expressiebomen. Dit is een datastructuur om een expressie in op te slaan. We hebben de theorie hierachter onderzocht. Met behulp van het Shunting-yardalgoritme kan een string worden omgezet in een expressieboom. De Reverse Polish Notation is een manier om een expressie op een eenduidige manier zonder haakjes op te schrijven. Dit komt overeen met het in post-order doorlopen van de expressieboom. We hebben zelf een aantal klassen geschreven in Python die een implementatie zijn van expressiebomen. Voorbeelden van berekeningen die we hiermee kunnen uitvoeren zijn differentiatie, numerieke integratie, negatie, substitutie, evaluatie en versimpeling.\\
\end{abstract}
\begin{figure}[h]
	\begin{tikzpicture}[->,>=stealth',shorten >=1pt,auto,node distance=3cm,thick,main node/.style={circle,draw,font=\sffamily\Large\bfseries}]
	
	\node[main node] (1) {$+$};
	\node[main node] (2) [below left of=1] {$\sin$};
	\node[main node] (4) [below right of=1] {$**$};
	\node[main node] (3) [below right of=4] {$3$};
	\node[main node] (5) [below left of=2] {$*$};
	\node[main node] (6) [below right of=5] {$y$};
	\node[main node] (7) [below left of=5] {$2$};
	\node[main node] (8) [below left of=3] {$+$};
	\node[main node] (9) [below left of=8] {$x$};
	\node[main node] (10) [below right of=8] {$1$};
	
	\path[every node/.style={font=\sffamily\small}]
	(1) edge node [left] {} (4)
	(1) edge node [left] {} (2)
	(2) edge node [left] {} (5)
	(5) edge node [right] {} (6)
	(5) edge node [left] {} (7)
	(4) edge node [right] {} (3)
	(4) edge node [left] {} (8)
	(8) edge node [left] {} (9)
	(8) edge node [right] {} (10);
	\end{tikzpicture}
\end{figure}
\newpage
\tableofcontents
\setcounter{page}{1}
\newpage

\section{Introductie}
Het rekenen met expressies is belangrijk voor het uitvoeren van ingewikkelde berekeningen. Een voorbeeld van een expressie is $(x+1)^3 + 3^2+ 6\sin(y)$. In een expressie komen getallen, variabelen en operatoren voor. Dit kunnen zowel binaire operatoren (zoals $+$ en $*$) als unaire operatoren (zoals $\sin$ en $\log$) zijn. Een voorbeeld van een programma dat kan rekenen met expressies is \textbf{Wolfram Mathematica}. Dit programma gebruikt hiervoor zogenaamde \textit{expressiebomen}. Dit is een datastructuur waarin de expressies worden opgeslagen. In dit verslag zullen we de theorie hierachter behandelen. Hierbij zullen we ook \textit{Reverse Polish Notation} (RPN) behandelen. Dit is een manier om een expressie op een eenduidige manier zonder haakjes weer te geven. Ook het \textit{Shunting-yardalgoritme} komt aan bod. Dit is een algoritme om een string om te zetten in een expressieboom. We hebben zelf in Python een aantal klassen geschreven die een implementatie zijn van expressiebomen. Voorbeelden van berekeningen die we hiermee kunnen doen zijn: het opslaan van een expresssie, waarden substitueren, het uitrekenen van een expressie, het combineren van twee expressies, negatie, versimpeling, differentiatie en integratie. In het gedeelte \textit{Theorie} zullen we de werking van expressiebomen, RPN en het Shunting-yardalgoritme uitleggen. In het deel \textit{Implementatie} leggen we uit hoe we de theorie hebben ge\"implementeerd in onze eigen code. We zullen de verschillende functionaliteiten uitleggen en een korte handleiding geven voor het gebruik van de code. In het gedeelte \textit{Conclusie en discussie} bespreken we de beperkingen van onze code.
\newpage
\section{Theorie}\label{theorie}%%%%%%%%%%%%%%%%%%%%%%%hier
De methodes die we geschreven hebben maken gebruik van de binaire boom structuur van expressie bomen. Een expressie boom heeft als knopen operatoren en constanten of variabelen als bladeren. Een eenvoudige expressie is bij voorbeeld $5+6$, de binaire operator $+$ is de stam en heeft als kinderen 5 en 6. Een operator mag ook een andere operator als kind hebben, zo zou in ons voorbeeld de operator $+$ ook de operator $*$ als rechterkind kunnen hebben. De $*$ knoop heeft dan ook twee kinderen, bijvoorbeeld 2 en 3. Om een geldige uitdrukking te zijn moeten de bladeren in de boom altijd een waarde representeren, dit mag ook een variabele zijn.

De boom wordt vanaf de bladderen gelezen, een of twee waarden worden door de bovenliggende knoop omgezet naar een nieuwe waarde en deze opnieuw met de bovenliggende totdat de stam van de boom bereikt is. Het volgt nu ook eenvoudig dat er geen haakjes nodig zijn, of een afspraak over de volgorde van links of rechts: een knoop heeft alleen interactie met zijn ouders of kinderen en waar een operator niet associatief is, zoals bij machtsverheffen, is dit in de knoop zelf verwerkt.


De methode \texttt{fromString} zet een expressie om in een expressie boom. Om dit te doen maakt het gebruik van postfix notatie en het shunting yard allogaritme. 
Postfix notatie ook wel `reversed Poolish notation', RPN, is een andere volgorde van het schrijven van een expressie. Hierbij worden de operatoren achter de waarden geschreven. Zo wordt $2+3$ in RPN geschreven als $2\ 3\ +$. Bij deze notatie is het van belang dat van te voren precies duidelijk is hoeveel argumenten een operator vraagt.
Bij de methode zijn geen haakjes nodig, bij de standaard infix schrijfwijze betekend $(4+5)*3$ is anders dan $4+5*3$ omdat we hebben afgesproken dat vermenigvuldigen voor optellen komt. RPN werkt op de volgende manier: \\
Bekijkt $4\ 5\ 3\ +\ *$. De eerste operator die we tegen komen is de plus, we kijken nu naar de eerste twee waardes, want plus neemt twee waardes. Dit zijn 4 en 5, die tellen we op tot 9. De volgende operator die we tegen komen is keer, ook deze neemt twee waardes, de waardes die nu vooraan staan zijn 9 en 3, deze vermenigvuldigen we. Deze expressie komt dus overeen met $(4+5)*3$. Merk op dat $4\ 5\ +\ 3\ *$ het zelfde is omdat we precies de zelfde stappen kunnen volgen. Het is dus niet nodig om alle operatoren achter aan te hebben staan.

Een alogaritme om infix notatie om te zetten in postfix is het shunting yard algoritme. Deze methode werkt met behulp van een stack voor operatoren. Het begint met lezen van de expresie aan de linkerkant. Aan de hand van de volgende stappen wordt de uitkomt van links naar rechts opgebouwd.
    \begin{itemize}
        \item Een getal komt op de uitkomst,
        \item Een functie komt op de uitkomst,
        \item Bij een operator $o$ wordt gekeken naar de stack:
        \begin{itemize}
            \item zolang de stack niet leeg is en de eerste operator op de stack heeft een lagere voorgang dan $o$, worden alle operatoren van de stack op de uitkomst gezet,
            \item dan zet $o$ op de stack.
        \end{itemize}
        \item Een linker haakje wordt op de stack geplaatst,
        \item Bij een rechter haakje worden alle operatoren van de stack gehaald en op de uitkomst geplaatst tot een rechter haakje wordt gevonden. Beide haakjes worden daarna vergeten.
        \item Als er niets meer is om te lezen maar de stack niet leeg, plaats de stack op de uitkomst.
    \end{itemize}

De complexiteit van dit algoritme is $O(n)$ afhankelijk van de lengte van de lengte van de input. Het algoritme hoeft per token maximaal een keer te lezen, een keer te schrijven op de uitput, een keer te schrijven op de stack en er van af te halen.
%%%%%%%%%%%%%%%%%%%%%%tot hier

\newpage
\section{Implementatie}
\subsection{Basiscode} %Arja
\subsection{De functionaliteiten van onze code}
In dit deel zullen we de verschillende berekeningen die onze code kan uitvoeren uitleggen. Bij elke functionaliteit geven we ook een korte handleiding voor het gebruik hiervan.

\subsection{Basiscode} %%%%%%%%%%%%%hier2
De basis van de code wordt gevormd door de class \texttt{Expression} en de subclasses die de verschillende knopen definiëren. In \texttt{Expression} staan methodes die voor alle expressies gedefinieerd zijn, zoals het maken van een string en het checken van gelijkheid, en de methode \texttt{fromString} welke een string inleest en een expressie boom teruggeeft. 

In \texttt{Expression} definiëren we geen intitalisatie, dit doen we in de subclasses. Alle knopen zijn subclasses van \texttt{Expression}, dus alle methode in deze hooft class werken om alle knopen. Hier in zijn ook optellen, vermenigvuldiging, logaritme, enzovoort gedefinieerd omdat we alle soorten expressie willen optellen, vermenigvuldigen. Deze werken allemaal op de zelfde manier: als we expressie $a$ bij expressie $b$ willen optellen, wordt en een optellingsknoop aangemaakt met als linker en rechter kind $a$ en $b$. Bij unaire functies wordt maar een kind toe gevoegd. Voor constanten en variabelen wordt een blad aangemaakt, deze knoop heeft alleen een waarde.
Dit heeft als voordeel dat een aantal ongeldige expressies niet mogelijk zijn om te maken: zo kan een optelling van een getal niet geschreven worden, de optellingsknoop vraagt namelijk twee argumenten.

\subsubsection{Het vertalen van een string naar een expressie boom}
De methode \texttt{fromString} zet een string om in een expressie boom waarop de rest van de methodes toe gepast kunnen worden. Dit vertalen gaan een drie stappen. De eerste is het omzetten van een string in een lijst met tokens. Dit gebeurd met de functie \texttt{tokenize}. De functie leest de gehele string en voor en na de operatoren $+,-,*,/$ en de linker en rechter haakjes staan worden spaties gezet. Tenslotte worden  geslist op spaties en wordt er voor gezorgd dat machtsverheffing en negatieve getallen juist worden gelezen.

De methode voert vervolgens het Shuting Yard algoritme uit zoals dit in \ref{theorie} is uitgelegd, met als resultaat een een lijst, output. De enige verschil tussen het standaard algoritme en de code is dat de code de numerieke waardes en de tokens die niet als operator of standaard wiskundige functie wordt herkent, als een \texttt{Constant} repertiefelijk \texxt{Variables}-knoop opslaat. De rede om deze meteen in een knoop om te zetten volgt uit de manier waarom de boom wordt opgebouwd.
%%%%%%%%%%%%%%%%%%%%%
De methode heeft nu een lijst, output, in RPN. Deze wordt nu omgezet in een expressie boom door het gedeelte dat de lijst leest met behulp van de standaard \texttt{eval} functie in Python. Opnieuw maakt het een stack aan en begint voor aan te lezen, alle waardes, constante of variabele knopen worden op de stack geplaatst. Zodra de methode een operator of functie tegen komt neemt het de bijbehorende een of twee waarden van de stack en evalueert dit.

De operatoren en de wiskundige functies zijn gedefinieerd als de bijbehorende binaire of unaire knoop met als kinderen de waardes waarover de operator wordt aangeroepen. Omdat de waardes waarover de methode evalueert expressie zijn gebruikt het de in \texttt{Expression} gedefinieerde functie: er wordt een nieuwe boom aangemaakt die de deel expressie gepresenteerd. Deze wordt weer op de stack geplaatst zodat een volgende evaluatie dit kan gebruiken. Als alle tokens in de output lijst zijn doorlopen blijft er een element in de stack over welke de voltooide expressie boom is.

%%%%%%%%%%%%%%%%%tot hier2
%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%
\subsubsection{Het vertalen van een expressie naar een string}
Een expressie kan worden omgezet naar een string. Dit gebeurt in de methode \texttt{\_\_str\_\_} in de klasse \texttt{BinaryNode}. Het maken van de string gebeurt op een recursieve manier. Er wordt telkens de \texttt{str}-methode aangeroepen van het linker- en het rechterkind. Deze roepen op hun beurt weer de \texttt{str}-methode aan van hun kinderen, enz. Wanneer een kind een \texttt{Constant} of een \texttt{Variable} is, zorgen de \texttt{str}-methoden van die klassen ervoor dat de juiste waarde als string wordt teruggegeven. Op die manier wordt de gehele string opgebouwd. \\
We hebben ervoor gezorgd dat er alleen maar haakjes worden geplaatst wanneer dit echt nodig is. Wanneer de volgorde waarin de operaties moeten worden uitgevoerd al volgen uit de prioriteit van de operatoren, kunnen haakjes worden weggelaten. Bijvoorbeeld bij de expressie $(2+3)*4$ zijn haakjes nodig maar bij $2+(3*4)$ kunnen ze worden weggelaten. Bij het implementeren hiervan gaat we uit van het volgende idee: wanneer er de binaire boom in een kind van een operator een operator voorkomt die een lagere prioriteit heeft, en deze is nog niet eerder ingesloten door haakjes, dan moeten er haakjes worden geplaatst.  \\
In afbeelding \ref{haakjesboom} zien we de boom die hoort bij $(2+3)*4*5$. Voor het onderste $*$-teken geldt dat er in zijn linkerkind een operator staat met lagere prioriteit: het $+$-teken. Om $2+3$ moeten dus haakjes worden geplaatst. Voor het bovenste $*$-teken geldt ook dat er in zijn linkerkind een $*$-teken voorkomt, maar dit is al eerder ingesloten door haakjes. Het is dus niet nodig om haakjes te plaatsen om $(2+3)*4$.\\
Om dit idee te implementeren geven we elke operator een prioriteit.
We geven $**$ prioriteit 1, $*$ en $/$ krijgen prioriteit 2 en $+$
en $- $ krijgen prioriteit 3. De functie \texttt{op\_nummer} geeft
ons de prioriteit van een operator. De functie \texttt{zoek\_op}
geeft ons de prioriteit van de operator met laagste prioriteit die nog niet eerder ingesloten is door haakjes. Hiervoor gebruiken we de variabele \texttt{open}. De string die van de expressie wordt van links naar rechts doorlopen. Als we een '(' tegenkomen, wordt \texttt{open} met $1$ wordt opgehoogd wanneer we een ')' tegenkomen wordt \texttt{open} met 1 verlaagd. De operatoren die nog niet zijn ingesloten door haakjes zijn nu degenen die we tegenkomen terwijl \texttt{open} gelijk aan $0$ is. We zoeken nu degene hiervan met de laagste prioriteit.\\\\
\textbf{Handleiding}\\
Hieronder een voorbeeld van hoe de code gebruikt kan worden\\\\
\texttt{>>> a = Expression.fromString('(1+x)*3*4')\\
		>>> print(a)\\
		(1 + x) * 3 * 4}

\begin{figure}
\begin{tikzpicture}[->,>=stealth',shorten >=1pt,auto,node distance=3cm,thick,main node/.style={circle,draw,font=\sffamily\Large\bfseries}]

\node[main node] (1) {$*$};
\node[main node] (2) [below left of=1] {$*$};
\node[main node] (3) [below right of=2] {$4$};
\node[main node] (4) [below right of=1] {$5$};
\node[main node] (5) [below left of=2] {$+$};
\node[main node] (6) [below right of=5] {$3$};
\node[main node] (7) [below left of=5] {$2$};

\path[every node/.style={font=\sffamily\small}]
(1) edge node [left] {} (4)
(1) edge node [left] {} (2)
(2) edge node [right] {} (3)
(2) edge node [left] {} (5)
(5) edge node [right] {} (6)
(5) edge node [left] {} (7);
\end{tikzpicture}
\caption{De expressieboom die hoort bij de expressie $(2+3)*4*5$}
\label{haakjesboom}
\end{figure}
\newpage
\subsubsection{Differentiatie}
Ons programma bevat de functionaliteit differenti\"eren. Hierbij wordt een expressie gedifferentieerd naar een gegeven variabele. Dit gebeurt door de methode \texttt{derivative} in de klasse \texttt{Expression}.  Het differenti\"eren gebeurt door middel van recursie. We hebben daarbij de onderstaande regels ge\"implementeerd. Hierbij is $x$ de variabele waarnaar we differenti\"eren, $f$ en $g$ zijn expressies en $k$ is een constante
\begin{itemize}
	\item De afgeleide van een constante is $0$
	\item De afgeleide van $x$ is $1$
	\item De afgeleide van elke andere variable is $0$
	\item $(f+g)' = f' +g'$
	\item $(f-g)' = f' -g'$
	\item $(f*g)' = f*g' + f'*g$
	\item $\left(\frac{f}{g}\right)' = \frac{g*f' - f*g'}{g^2}$
	\item $(f^k)' = k*f^{k-1}*f'$
	\item $\left(\sin(f)\right)' = \cos(f)*f'$
	\item $\left(\cos(f)\right)' = -\sin(f)*f'$
	\item $\left(\tan(f)\right)' = (\tan^2(f)+1)*f'$
	\item $\left(\log(f)\right)' = \frac{f'}{f}$
	\item $(f^g)' = f^{g}\left[g'\log(f) + g\frac{f'}{f}\right]$
\end{itemize}
Op deze manier breken we het probleem op in steeds kleinere stukken en kunnen we van veel verschillende uitdrukkingen de afgeleide nemen. In het laatste punt zien we dat we ook de afgeleide kunnen nemen van een expressie waarbij de variabele waarnaar we willen differenti\"eren in een exponent staat zoals bij $x^x$. Hiervoor is logaritmisch differenti\"eren nodig. De regel volgt uit het volgende\\
Zij $h(x) = f(x)^{g(x)}$. Definieer $l(x) = \log(h(x))$. \\ Enerzijds geldt dan dat $l'(x) = \frac{h'(x)}{h(x)}$. Anderzijds geldt $l'(x) = \left[\log(f(x)^{g(x)})\right]' = \left[g(x)\log(f(x))\right]' = g'(x)\log(f(x)) + g(x)\frac{f'(x)}{f(x)}$.\\
Hieruit volgt $h'(x) = f(x)^{g(x)}\left[g'(x)\log(f(x)) + g(x)\frac{f'(x)}{f(x)}\right]$.\\\\
\textbf{Handleiding}\\
De methode \texttt{derivative} heeft als input de expressie en de variabele waarnaar gedifferentieerd moet worden. Deze variabele moet als string worden ingevoerd. Het resultaat is een nieuwe expressie. De bestaande expressie blijft dus onveranderd. Hieronder een voorbeeld\\\\
\texttt{>>> a = Expression.fromString('2*x**3 + x')\\
		>>> print(a.derivative('x'))\\
		2 * 3 * x ** 2 * 1 + x ** 3 * 0 + 1}
	
\textbf{Beperkingen}\\
De afgeleide die we krijgen is vaak onnodig ingewikkeld, zoals in het voorbeeld te zien is. Daarom wordt aangeraden om na het nemen van de afgeleiden de methode \texttt{simplify} toe te passen.

\subsubsection{Numerieke integratie}
De klasse \texttt{Expression} bevat de methode \texttt{num\_integration} waarmee we een expressie numeriek kunnen integreren. Hiervoor gebruiken we een Riemannsom. Stel dat we voor een bepaalde expresse $f$ die afhangt van $x$ willen bepalen
$$I = \int_{a}^{b}f(x)\text{d}x$$ 
We delen het interval $[a,b]$ op in $n$ delen van breedte $\Delta x = \frac{b-a}{n}$. Hierdoor ontstaan er $n+1$ punten $x_i =i\left(\frac{b-a}{n}\right)+a$ met $i =0,1,...,n$. We benaderen de integraal met
$$I \approx \sum_{i=0}^{n-1}f(x_i)\Delta x$$\\\\
\newpage
\textbf{Handleiding}\\
We kunnen deze functionaliteit aanroepen met \texttt{f.num\_integration(a,b,'x',n)}. We kunnen dus kiezen in hoeveel punten we het interval willen verdelen. Meer punten zal leiden tot een kleinere fout in het antwoord maar dit kost wel een langere rekentijd. In onderstaande voorbeeld bepalen we $\int_{0}^{1}(x^2+1)\text{d}x$ met 10000 punten. Het exacte antwoord is $\frac{4}{3}$ dus de fout is ongeveer $5\cdot 10^{-5}$.\\\\
\texttt{>>> a = Expression.fromString('x**2 + 1')\\
		>>> print(a.num\_integration(0,1,'x',10000))\\
		1.3332833349999427}\\\\
We kunnen ook expressies integreren die meer dan \'e\'en variabele bevatten. Aan de andere variabelen moet dan via een dictionary een waarde worden toegekend. Hieronder zien we hoe dat in zijn werk gaat.\\
\texttt{>>> a = Expression.fromString('x+y')\\
		>>> print(a.num\_integration(0,1,'x',1000,\{'y':1\}))\\
		1.4995000000000005}\\\\
\textbf{Complexiteit}\\
We hebben de complexiteit bepaald van de methode \texttt{num\_integration}. Hiervoor hebben we de methode toegepast op de integraal\\ $\int_{0}^{1}3x^2+4x+1 \text{d}x$ voor verschillende waarden van $n$. In onderstaande tabel zijn voor verschillende $n$ de tijd en de fout in het antwoord te zien. Het exacte antwoord is $4$ dus we hebben de fout gedefineerd als het absolute verschil met 4. Voor elke $n$ hebben we $10$ keer de meting uitgevoerd en hiervan het gemiddelde genomen.
\begin{table}[h]
	\centering
	\begin{tabular}{r|c|c}
	n&T (sec.)&fout\\ \hline
	$1$&$1.9 \cdot 10^{-3}$&$3$\\
	$10$&$3.9 \cdot 10^{-3}$&$0.34$\\
	$100$&$2.3 \cdot 10^{-2}$&$3.5 \cdot 10^{-2}$\\
	$1000$&$0.21$&$3.5 \cdot 10^{-3}$\\
	$10.000$&$2.0$&$3.5 \cdot 10^{-4}$\\
	$100.000$&$21$&$3.5 \cdot 10^{-5}$\\
	\end{tabular}
	\label{hai}
\end{table}
We zien dat de complexiteit van $\mathcal{O}(n)$ is. Dit lijkt logisch aangezien we voor elk punt dat we toevoegen de expressie moeten evalueren, het resultaat hiervan moet vermenigvuldigen met $\Delta x$ en $x_i$ moeten ophogen. Het aantal operaties schaalt dus lineair met het aantal punten. We zien dat ook de fout daalt met $\mathcal{O}(n)$.
		
\subsubsection{Negatie}
De overload \texttt{\_\_neg\_\_} voor negatie van expressiebomen werkt door de hele boom te vermenigvuldigen met $(-1)$. In feite wordt er dus een \texttt{BinaryNode} aangemaakt met als operator \texttt{*} en als linkertak $(-1)$ en rechtertak de oorspronkelijke boom.

\textbf{Handleiding}\\
Negatie van een expressieboom werkt door simpelweg een minteken voor de expressieboom te zetten. Bij voorbeeld: \\\\
\texttt{>>> a = Expression.fromString('2 + 1')\\
		>>> b =-a\\
		>>> print(a)\\
		2+1\\
		>>> print(b)\\
		 (-1)*(2+1)}

\subsubsection{Gelijkheid van expressiebomen}
Of twee expressiebomen dezelfde berekening voorstellen, kan getest worden door middel van de methode \texttt{\_\_eq\_\_} van de klasse \texttt{Expression}. De code voor \texttt{\_\_eq\_\_} loopt recursief langs alle \texttt{BinaryNode}s totdat er een \texttt{MonoNode}, \texttt{Constant} of \texttt{Variables} wordt bereikt. Nodes van deze klassen kunnen eenvoudig worden vergeleken. Voor \texttt{BinaryNode}s is de code iets ingewikkelder. Laten we de expressiebomen die vergeleken worden \texttt{self} en \texttt{other} noemen. Om te beginnen gaat de code er van uit dat twee \texttt{BinaryNode}s alleen gelijk kunnen zijn als zij dezelfde operator bevatten. Daarna wordt er onderscheid gemaakt tussen commutatieve en niet-commutatieve operatoren. Voor commutatieve operatoren wordt er gekeken naar of de linkerkanten van \texttt{self} en \texttt{other} gelijk aan elkaar zijn en de rechterkanten ook, \'of dat de linkerkant van \texttt{self}  gelijk is aan de rechterkant van \texttt{other} en de rechterkant van \texttt{self} gelijk is aan de linkerkant van \texttt{other}. Immers, neem het voorbeeld $2 * 3 = 3 * 2$. Voor niet-commutatieve operatoren wordt er alleen gekeken naar of de linkerkanten van \texttt{self} en \texttt{other} gelijk aan elkaar zijn en de rechterkanten ook. Immers, $2**3 \neq 3**2$ dus geldt het tweede geval van gelijkheid zoals omschreven voor commutatieve operatoren niet. Dat de code recursief is blijkt uit dat de linker- en/of rechterkant van een \texttt{BinaryNode} niet noodzakelijk een \texttt{MonoNode}, \texttt{Constant} of \texttt{Variables} is, maar ook meer \texttt{BinaryNode}s kan bevatten waarvoor het proces herhaald moet worden. Om daarvoor te zorgen wordt de functie \texttt{\_\_eq\_\_} (met het teken \texttt{==}) opnieuw aangeroepen om bij voorbeeld \texttt{self.lhs} met \texttt{other.rhs} te vergelijken, waarbij \texttt{lhs} staat voor "left-hand side", ofwel de linkertak van \texttt{self}/\texttt{other}, en \texttt{rhs} voor "right-hand side", ofwel de rechtertak van de knoop.\\\\

\textbf{Handleiding}\\
De methode \texttt{\_\_eq\_\_} kan toegepast worden op twee bomen die gemaakt zijn met de functie \texttt{fromString}. Bij voorbeeld: \\\\
\texttt{>>> a = Expression.fromString('2 + 1')\\
		>>> b = Expression.fromString('1 + 2')\\
		>>> print(a==b)
		True }

\textbf{Beperkingen}\\
De code kan niet herkennen dat b.v. $-1+2$ en $2-1$ hetzelfde zijn. Dit zou lastig te implementeren zijn omdat je iets zou kunnen doen zoals: als \texttt{self.op\_symbol==’+’} en \texttt{other.op\_symbol==’-‘} (of andersom), kijk dan of de linkerkant van de ene boom gelijk is aan de negatie van de andere boom, en de rechterkant gelijk aan de linkerkant van de ander. Het probleem zit dan in de negatie. De negatie van een boom geeft $(-1)$ keer de oorspronkelijke boom, maar de code herkent de gelijkheid van $-1$ en $(-1)*1$ niet. Dit zou ge\"implementeerd kunnen worden, maar dit wordt ingewikkeld omdat 1 weer opgebouwd kan worden van bij voorbeeld $0+1$, dus moet ook de gelijkenis van $(-1)*(0+1)$ of nog grotere bomen herkend worden. Dit zou opgelost kunnen worden door een versimpeling hiervoor te implementeren.

\subsubsection{Het numeriek uitrekenen van een expressie} 
Als men de numerieke waarde van een expressie zou willen berekenen, ofwel een expressie 'evalueren', dan zou dat kunnen met de methode \texttt{evaluate} van de klasse \texttt{Expression}. Deze methode neemt als input een dictionary die waarden toekent aan de variabelen van de expressie. Als er geen variabelen in de expressieboom zitten, dan hoeft er geen input gegeven te worden en gaat de code er vanuit dat er een lege dictionary \texttt{\{\}} wordt meegegeven. Ook maakt de methode gebruik van de ingebouwde functionaliteit \texttt{eval} van Python, die ook als input een dictionary moet krijgen voor onbekende variabelen. De methode \texttt{evaluate} loopt de expressieboom af totdat hij bij een knoop van het type \texttt{Constant}, \texttt{Variables}, \texttt{LogNode}, \texttt{SinNode}, \texttt{CosNode} of \texttt{TanNode} komt. Bij zo'n knoop berekent hij eevoudig de numerieke waarde. Voor een  \texttt{Constant} of \texttt{Variables} knoop houdt dat in het invullen van de waarde (voor variabelen uit de bijgevoegde dicionary gehaald) in \texttt{eval}. Voor een \texttt{LogNode}, \texttt{SinNode}, \texttt{CosNode} of \texttt{TanNode} staat een aparte \texttt{evaluate} methode in zijn eigen klasse. Vervolgens voegt de code de uitkomsten samen voor alle \texttt{BinaryNode}s van onder naar boven met behulp van \texttt{eval}.\\
Ook gedeeltelijke evaluatie wordt ondersteund. Dit kan uitgevoerd worden met de methode \texttt{part\_evaluate} die voortbouwt op de methode \texttt{evaluate}. Hiermee kunnen sommigen van de variabelen een waarde toegekend krijgen via de dictionary, en deze worden ingevuld, terwijl anderen onbepaald mogen blijven. Vervolgens worden gedeelten van de boom zonder onbepaalde variabelen vereenvoudigd. Als geen van de variabelen in de dictionary voorkomen, dan geeft \texttt{part\_evaluate} de oorspronkelijke expressieboom terug. Het idee van de code is dat er een nieuwe string \texttt{newstring} wordt aangemaakt waarin de gedeeltelijk uitgerekende expressie wordt opgeslagen. Voor een \texttt{BinaryNode} wordt recursief de functie \texttt{part\_evaluate} losgelaten op de linkerkant en rechterkant, en de uitkomst wordt toegevoegd aan de string met de operator van de \texttt{BinaryNode} ertussen. Een knoop van het type \texttt{Constant} wordt eenvoudigweg toegevoegd aan de string en een knoop van het type \texttt{Variables} ook, eventueel met de variabele vervangen door zijn waarde als deze bekend is uit de dictionary. Een \texttt{MonoNode} wordt toegevoegd als een knoop van hetzelfde type, maar met de expressie binnen de haakjes gedeeltelijk uitgewerkt, voor zo ver mogelijk. Uiteindelijk wordt er een nieuwe expressieboom gemaakt door middel van de methode \texttt{fromString} en met als input de string \texttt{newstring}. Als laatste wordt er voor gezorgd dat de boom zo vereenvoudigd mogelijk teruggegeven wordt.

\textbf{Handleiding}\\
Hieronder volgen twee voorbeelden van hoe de methode \texttt{evaluate} kan worden toegepast op een expressieboom. Het is belangrijk om op te merken dat onze code alle input van het type \texttt{str} (strings) interpreteert als een variabele. Zo zal de waarde van een variabele \texttt{a}, aan welke in \texttt{\_\_main\_\_} een waarde is toegekend, niet herkend worden in de functie \texttt{evaluate}. De waarde zal opnieuw moeten worden meegegeven in de dictionary. \\ 

\texttt{>>> a = Expression.fromString('2 + 1')\\
		>>> print(a.evaluate())\\
		3 \\
		>>> b = Expression.fromString('x+sin(pi)')\\
		>>> print(b.evaluate(\{'x':1,'pi':math.pi\}))\\
		1}\\

Een paar voorbeelden van hoe \texttt{part\_evaluate} werkt: \\\\
\texttt{>>> a = Expression.fromString('x+y')\\
		>>> print(a.part\_evaluate({'x':1}))\\
		1+y \\
		>>> b = Expression.fromString('log(x**y+2/z)')\\
		>>> print(b.part\_evaluate())\\
		log(x**y+2/z)}\\
		
\subsubsection{Sympify}
De methode \texttt{symplify} is een bedoelt om expressie bomen te verkleinen. Hierin worden zoveel mogelijke gevallen van gevallen getest die de expressie boom kunnen vereenvoudigen. Zo kunnen numerieke waardes uitgerekend worden en vervangen door een enkele constante knoop welke de geëvalueerde waarde bevat. Een andere vereenvoudiging die wordt toe gepast is het optellen van nul en vermenigvuldigen van een en een variabele: dit wordt vervangen door de variabele. Net zo wordt vermenigvuldigen met nul door een constante met nul vervangen.

Dit zijn tot nu toe de enige gevallen die de methode behandeld. De methode kan worden uitgebreid met gelijkheden als $ax+bx=(a+b)x$ waarbij $(a+b)$ geëvalueerd kan worden, en gelijkheden als $log(0) = 1$ en de goniometrische gelijkheden.

We merken op dat \texttt{symplify} de expressie zelf aanpast en die een versimpelde functie terug geeft. Hiervoor is gekozen omdat de versimpelde functie kleiner en dus minder intensief is om mee te rekenen. 

\subsubsection{Solve}
De klasse \texttt{Expression} bevat een methode \texttt{solve} die het nulpunt zoekt van een expressie met een variabele. 
Het is een klassiek vraagstuk in de wiskunde om het vinden van een nulpunt van een functie. Dit kan op twee manieren gedaan worden: algebraïsch en numeriek. Algebraïsch oplossen is voor computers ingewikkeld, hierbij komt vaak een complete trukendoos bij kijken en een intuïtie wanneer een bepaald hulpmiddel nuttige vorderingen oplevert. Een zeer geavanceerd programma zoals \textbf{Wolfram Mathematica} kan ook algebraïsch vergelijkingen oplossen. Numeriek oplossen van vergelijkingen is eenvoudiger te implementeren en er zijn veel variaties hierop. 

De Newton-Raphson is een eenvoudig voorbeeld hiervan. Deze methode werkt voor veel continue differentieerbare functies. Stel $f$ is de functie waarvan het nulpunt bepaald moet worden en $f'$ zijn afgeleide. Er is een begin punt nodig vanwaar de functie opzoek aat naar het nulpunt, het punt $x_0$. Het is niet nodig dat dit punt in de buurt van het nulpunt ligt, maar het maakt de methode wel sneller als dat zo is. Dit is een iteratief proces waarbij een volgend nulpunt benaderd worden met de vorige op de volgende manier:
    \begin{equation}
        x_{k+1} = x_k + \frac{f(x_k)}{f'(x_k)}
    \end{equation}
Wanneer het verschil tussen $x_k$ kleiner is dan de tolerance stop het iteratief proces. 


\textbf{Handleiding} \\
De methode \texttt{solve} moet worden aangeroepen met een expressie en de variabele waarnaar de expressie moet worden opgelost, deze variabele moet als string worden ingevoerd.

De methode heeft als standaardinstelling dat het het nulpunt van de expressie zoekt, door als tweede waarde een getal $a$ mee te geven zoekt het de een punt waar de expressie $a$ is, dit kan ook door \texttt{y = a} in te vullen.

De methode begint de iteratie standaard met $x_0 = 0$. Een ander beginpunt $b$ kan worden opgegeven als \texttt{x0 = b}. Dit kan de mogelijkheid geven om toch nog een nulpunt te vinden als dit in eerste instantie niet gelukt is, of een ander nulpunt.

Standaard stopt de methode als het verschil tussen $f(x_k)$ en $0$ of $a$ minder dan $10^{-5}$ is. Door \texttt{tolerance} $=\epsilon$  aan de methode mee te geven veranderd de tolerantie naar $\epsilon$. 

Als laatste is er de optie om het aantal iteraties te bepalen, standaard zijn dit er duizend. Door \texttt{n = N} mee te geven stop de functie na maximaal N iteraties. Ook kan \texttt{n = None}, dat is er geen bovengrens van iteraties en blijft de functie itereren tot de tolerantie bereikt is.

Als de functie geen nulpunt kan vinden, geeft de methode een foutmelding waarin alle variabele worden gerepresenteerd.

Hieronder twee voorbeelden \\\\
\texttt{>>> f = Expression.fromString('2*x**3 + x', n = None)\\
		>>> print(f.solve('x'))\\
		}

\texttt{>>> f = Expression.fromString('2*x**4 + x')\\
		>>> print(f.solve('x', x$_0$ = 5))\\
		}
		
\textbf{Beperkingen}\\
De beperkingen van deze methode liggen in de beperkingen van de Newton methode: niet alle nulpunten van een functie kunnen gevonden worden omdat de methode daar niet stabiel is. Dit ligt aan de afgeleide van de functie in dit punt.

Een verbetering kan zijn het toevoegen van meerdere algoritmes om nulpunten te vinden, zoals de bisectie methode die ook een nulpunt kan vinden als dat Newton onstabiel is door de afgeleide. Deze andere methodes hebben hun eigen nadelen. 

\newpage
\section{Conclusie en discussie}
We hebben gezien dat we met behulp van het Shunting-yardalgoritme expressies kunnen omzetten in een expressieboom. Vervolgens kunnen we met de expressies gaan rekenen. We hebben een aantal klassen geschreven in Python en hiermee kunnen we al aardig wat verschillende berekeningen en operaties uitvoeren. Er zijn echter ook beperkingen aan onze code. Expressies zijn soms onnodig ingewikkeld. Vooral bij het resultaat van differenti\"eren is dit het geval. De methode \texttt{simplify} biedt hier een uitkomst, maar nog niet alle gevallen worden hiermee opgelost. Ook het vergelijken van bijvoorbeeld $-1+2$ met $2-1$ gaat nog niet helemaal goed. Verder zouden er nog meer functionaliteiten toegevoegd kunnen worden. Een voorbeeld is een methode die de primitieve van een expressie bepaalt.

\section{Taakverdeling}
We hebben de taken als volgt verdeeld:\\\\
\textbf{Isabel}\\
Het schrijven van de code voor
\begin{itemize}
\item Het vertalen van een expressie naar een string en het gebruik van haakjes
\item Differentiatie
\item Numerieke integratie
\end{itemize}
en het schrijven van de overeenkomstige delen van verslag. Verder de samenvatting, inleiding en conclusie/discussie.

\textbf{Nina}\\
Het schrijven van de code voor
\begin{itemize}
\item Overload van de gelijkheidsoperator
\item Het berekenen van de numerieke waarde van een expressie (\texttt{evaluate} en \texttt{part\_evaluate})
\item Negatie
\item Een deel van \texttt{tokenize} voor het begrijpen van negatieve getallen in de input
\end{itemize}
en het schrijven van de overeenkomstige delen van verslag (met uitzondering van het stukje code in \texttt{tokenize}).

%%%%%%%%%%%%%%%%%%%%%%%
\textbf{Arja}\\
Het schrijven van de code voor
\begin{itemize}
    \item Het aanpassen van \texttt{fromString} zodat andere operatoren op de juiste manier verwerkt kunnen worden
    \item De unaire operatoren
    \item De \texttt{symplify} methode
    \item De \texttt{solve} methode
    \item Ongelijkheid van bomen juist definiëren
\end{itemize}
en het schrijven van de overeenkomstige delen in het verslag, samen met de uitleg en gebruik van \texttt{fromString} en het stuk theorie. 
%%%%%%%%%%%%%%%%%%%%%%
\newpage

\begin{thebibliography}{1}	
   \bibitem Wikipedia over de onderwerpen Reverse Polish Notation, Shuting Yard alogaritme en Newton-Rhapson methode.
	\bibitem{boek}
    Uri M. Ascher, Chen Greif:
    \emph{A First Course in Numerical Methods}, 2011, Saim
\end{thebibliography}


\end{document} 
