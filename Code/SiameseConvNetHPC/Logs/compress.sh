for file in *.txt
do
	col -b < "$file" > temp.txt
	mv temp.txt "$file"
done
