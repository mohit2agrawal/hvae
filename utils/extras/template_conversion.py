input_sents_file = "input_sentences.txt"
input_labels_file = "input_templates.txt"

with open('templates_pos.txt') as tpl_f, open(input_sents_file,
                                              'w') as inp_sf, open(
                                                  input_labels_file, 'w'
                                              ) as inp_lf:

    write_sent = True
    for template in tpl_f:
        if not template.strip():
            write_sent = True
            inp_lf.write('\n')
            continue
        if write_sent:
            inp_sf.write(template)
            write_sent = False
        else:
            inp_lf.write(template)
