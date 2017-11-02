for f in calBoard_l*.png
do
	mv $f `echo $f | sed 's/_l_/_r_/'`;
done
