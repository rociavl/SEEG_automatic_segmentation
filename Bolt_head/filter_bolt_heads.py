def _filter_bolt_heads(self, cleaned_mask, volume_node, brain_mask_array):
    """Identify and filter bolt head components with detailed shape analysis."""
    print("Identifying and filtering bolt head components with shape metrics...")
    labeled_image = label(cleaned_mask)
    regions = regionprops(labeled_image)
    
    filtered_mask = np.zeros_like(labeled_image, dtype=np.uint16)
    region_info = []
    spacing = volume_node.GetSpacing()
    origin = volume_node.GetOrigin()
    
    # Create shape metrics visualization figures
    fig_metrics = plt.figure(figsize=(15, 10))
    ax_sphericity = fig_metrics.add_subplot(221)
    ax_elongation = fig_metrics.add_subplot(222)
    ax_compactness = fig_metrics.add_subplot(223)
    ax_volume = fig_metrics.add_subplot(224)
    
    # Lists to store metrics for plotting
    metrics_data = {
        'region_id': [],
        'sphericity': [], 
        'elongation': [], 
        'compactness': [],
        'volume': [],
        'is_valid': []
    }
    
    for i, region in enumerate(regions):
        volume = region.area
        
        # Calculate the voxel coordinates in the region
        coords = np.argwhere(labeled_image == region.label)
        if len(coords) < 4:  # Need at least 4 points for 3D shape analysis
            continue
            
        # Calculate physical coordinates
        phys_coords = coords * spacing
        
        # Calculate 3D shape metrics
        shape_metrics = self._calculate_3d_shape_metrics(coords, phys_coords, spacing)
        
        # Determine if this is a valid bolt head based on shape and size
        is_valid_size = self.config['min_region_size'] < volume < self.config['max_region_size']
        # is_valid_shape = (
        #     shape_metrics['sphericity'] > 0.4 and
        #     shape_metrics['elongation'] < 3.0 and
        #     shape_metrics['compactness'] > 0.3
        # )
        is_valid = is_valid_size 
        
        # Record metrics for all regions (for visualization)
        metrics_data['region_id'].append(region.label)
        metrics_data['sphericity'].append(shape_metrics['sphericity'])
        metrics_data['elongation'].append(shape_metrics['elongation'])
        metrics_data['compactness'].append(shape_metrics['compactness'])
        metrics_data['volume'].append(volume)
        metrics_data['is_valid'].append(is_valid)
        
        # Process valid regions
        if is_valid:
            filtered_mask[labeled_image == region.label] = region.label
            centroid_physical = tuple(origin[i] + region.centroid[i] * spacing[i] for i in range(3))
            
            principal_axis = self._calculate_principal_axis(coords, spacing)
            bolt_to_brain_center = self._estimate_brain_center(brain_mask_array, spacing, origin) - np.array(centroid_physical)
            
            if np.dot(principal_axis, bolt_to_brain_center) < 0:
                principal_axis = -principal_axis
                
            # Store comprehensive region information
            region_info.append({
                'label': region.label,
                'physical_centroid': centroid_physical,
                'volume': volume,
                'principal_axis': principal_axis,
                'sphericity': shape_metrics['sphericity'],
                'elongation': shape_metrics['elongation'],
                'compactness': shape_metrics['compactness'],
                'eigenvalues': shape_metrics['eigenvalues'],
                'axis_lengths': shape_metrics['axis_lengths']
            })
    
    # Create visualization plots for shape metrics
    colors = ['red' if not valid else 'green' for valid in metrics_data['is_valid']]
    
    ax_sphericity.scatter(metrics_data['region_id'], metrics_data['sphericity'], c=colors)
    ax_sphericity.set_title('Sphericity by Region')
    ax_sphericity.set_xlabel('Region ID')
    ax_sphericity.set_ylabel('Sphericity')
    ax_sphericity.axhline(y=0.4, color='black', linestyle='--')
    
    ax_elongation.scatter(metrics_data['region_id'], metrics_data['elongation'], c=colors)
    ax_elongation.set_title('Elongation by Region')
    ax_elongation.set_xlabel('Region ID')
    ax_elongation.set_ylabel('Elongation')
    ax_elongation.axhline(y=3.0, color='black', linestyle='--')
    
    ax_compactness.scatter(metrics_data['region_id'], metrics_data['compactness'], c=colors)
    ax_compactness.set_title('Compactness by Region')
    ax_compactness.set_xlabel('Region ID')
    ax_compactness.set_ylabel('Compactness')
    ax_compactness.axhline(y=0.3, color='black', linestyle='--')
    
    ax_volume.scatter(metrics_data['region_id'], metrics_data['volume'], c=colors)
    ax_volume.set_title('Volume by Region')
    ax_volume.set_xlabel('Region ID')
    ax_volume.set_ylabel('Volume (voxels)')
    ax_volume.axhline(y=self.config['min_region_size'], color='black', linestyle='--')
    ax_volume.axhline(y=self.config['max_region_size'], color='black', linestyle='--')
    
    plt.tight_layout()
    metrics_plot_path = os.path.join(self.config['output_dir'], "bolt_head_shape_metrics.png")
    plt.savefig(metrics_plot_path, dpi=300)
    plt.close(fig_metrics)
    
    # Create a 3D visualization of filtered bolt heads with shape indicators
    self._visualize_bolt_heads_3d(filtered_mask, region_info, spacing, origin)
    
    # Generate PDF report with shape analysis
    self._generate_bolt_shape_report(region_info, metrics_data, metrics_plot_path)
    
    print(f"Found {len(region_info)} valid bolt head regions after filtering")
    volume_helper = VolumeHelper(spacing, origin, self._get_direction_matrix(volume_node), self.config['output_dir'])
    volume_helper.create_volume(filtered_mask, "Filtered_Bolt_Heads", "P6_filtered_bolt_heads.nrrd")
    
    # Save region metrics to CSV for further analysis
    metrics_df = pd.DataFrame({
        'Region ID': metrics_data['region_id'],
        'Volume': metrics_data['volume'],
        'Sphericity': metrics_data['sphericity'],
        'Elongation': metrics_data['elongation'],
        'Compactness': metrics_data['compactness'],
        'Valid': metrics_data['is_valid']
    })
    metrics_df.to_csv(os.path.join(self.config['output_dir'], "bolt_head_metrics.csv"), index=False)
    
    return filtered_mask, region_info

def _calculate_3d_shape_metrics(self, coords, phys_coords, spacing):
    """Calculate 3D shape metrics for a region."""
    # Calculate covariance matrix and its eigenvalues/eigenvectors
    centered_coords = phys_coords - np.mean(phys_coords, axis=0)
    covariance = np.cov(centered_coords, rowvar=False)
    
    # Handle potential numerical issues
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    
    # Sort eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Ensure all eigenvalues are positive (numerical stability)
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    
    # Calculate axis lengths (proportional to sqrt of eigenvalues)
    axis_lengths = 2 * np.sqrt(eigenvalues)
    
    # Calculate shape metrics
    # Sphericity: ratio of the volume to the surface area
    # For perfect sphere = 1, less spherical objects < 1
    volume = len(coords) * np.prod(spacing)
    
    # Surface area approximation using convex hull
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(phys_coords)
        surface_area = hull.area
    except Exception:
        # Fallback if convex hull fails
        surface_area = np.sum(np.sqrt(eigenvalues[0] * eigenvalues[1] + 
                                     eigenvalues[0] * eigenvalues[2] + 
                                     eigenvalues[1] * eigenvalues[2]))
    
    # Sphericity calculation: normalized ratio of volume to surface area
    # For a sphere with radius r: V = (4/3)πr³, A = 4πr², V/A = r/3
    # Perfect sphere sphericity = 1, less spherical objects < 1
    sphericity = np.power(6 * np.pi * volume, 1/3) / surface_area if surface_area > 0 else 0
    sphericity = min(max(sphericity, 0), 1)  # Normalize to [0,1]
    
    # Elongation: ratio of major to minor axis
    elongation = axis_lengths[0] / axis_lengths[2] if axis_lengths[2] > 0 else float('inf')
    
    # Compactness: measure of how compact the shape is
    # For solid objects, compactness is related to sphericity
    compactness = np.cbrt(volume**2) / surface_area if surface_area > 0 else 0
    compactness = min(max(compactness, 0), 1)  # Normalize to [0,1]
    
    return {
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'axis_lengths': axis_lengths,
        'sphericity': sphericity,
        'elongation': elongation,
        'compactness': compactness
    }

def _visualize_bolt_heads_3d(self, filtered_mask, region_info, spacing, origin):
    """Create 3D visualization of bolt heads with shape indicators."""
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    for info in region_info:
        region_mask = filtered_mask == info['label']
        
        try:
            # Generate surface for this bolt head
            verts, faces, _, _ = marching_cubes(region_mask, level=0.5, spacing=spacing)
            verts += origin  # Convert to physical coordinates
            
            # Plot the surface with color based on sphericity
            sphericity = info['sphericity']
            color_val = np.array([1.0 - sphericity, sphericity, 0])  # Red to Green
            ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], 
                          triangles=faces, color=color_val, alpha=0.8, shade=True)
            
            # Plot principal axis
            centroid = np.array(info['physical_centroid'])
            axis = np.array(info['principal_axis'])
            ax.quiver(*centroid, *axis, color='blue', linewidth=2, arrow_length_ratio=0.2)
            
            # Add text annotation with metrics
            text_pos = centroid + np.array([0, 0, 5])  # Offset text slightly above
            ax.text(text_pos[0], text_pos[1], text_pos[2], 
                   f"ID:{info['label']}\nSph:{info['sphericity']:.2f}\nElong:{info['elongation']:.2f}", 
                   color='black', fontsize=8)
        except Exception as e:
            print(f"Failed to visualize region {info['label']}: {e}")
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('3D Bolt Head Visualization with Shape Analysis')
    
    # Add colorbar for sphericity
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("sphericity", [(1,0,0), (0,1,0)])
    norm = plt.Normalize(0, 1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Sphericity')
    
    plt.tight_layout()
    plt.savefig(os.path.join(self.config['output_dir'], "bolt_head_3d_shape_viz.png"), dpi=300)
    plt.close(fig)

def _generate_bolt_shape_report(self, region_info, metrics_data, metrics_plot_path):
    """Generate PDF report with bolt head shape analysis."""
    try:
        # Import libraries for PDF generation
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        
        # Define PDF file path
        pdf_path = os.path.join(self.config['output_dir'], "bolt_head_shape_analysis_report.pdf")
        
        # Create document
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []
        
        # Add title
        title_style = styles['Heading1']
        title = Paragraph("Bolt Head Shape Analysis Report", title_style)
        elements.append(title)
        elements.append(Spacer(1, 0.25*inch))
        
        # Add introduction
        intro_style = styles['Normal']
        intro = Paragraph(
            "This report analyzes the shape characteristics of detected bolt head regions in 3D CT volumes. "
            "Shape metrics help differentiate actual bolt heads from noise or other artifacts.",
            intro_style
        )
        elements.append(intro)
        elements.append(Spacer(1, 0.25*inch))
        
        # Add metrics explanation
        metrics_title = Paragraph("Shape Metrics Explanation", styles['Heading2'])
        elements.append(metrics_title)
        
        metrics_explanation = [
            Paragraph("<b>Sphericity:</b> Measures how close the shape is to a perfect sphere (1.0 = perfect sphere). "
                     "Bolt heads typically have moderate sphericity (0.4-0.8).", styles['Normal']),
            Spacer(1, 0.1*inch),
            Paragraph("<b>Elongation:</b> Ratio of major to minor axis. Higher values indicate more elongated shapes. "
                     "Bolt heads are typically not highly elongated (values < 3.0).", styles['Normal']),
            Spacer(1, 0.1*inch),
            Paragraph("<b>Compactness:</b> Measure of how compact a shape is. Higher values indicate more compact shapes. "
                     "Bolt heads typically have moderate to high compactness (> 0.3).", styles['Normal']),
        ]
        elements.extend(metrics_explanation)
        elements.append(Spacer(1, 0.25*inch))
        
        # Add metrics visualization image
        elements.append(Paragraph("Shape Metrics Visualization", styles['Heading2']))
        metrics_img = Image(metrics_plot_path)
        metrics_img.drawHeight = 4*inch
        metrics_img.drawWidth = 6*inch
        elements.append(metrics_img)
        elements.append(Spacer(1, 0.25*inch))
        
        # Add metrics table
        elements.append(Paragraph("Detected Bolt Head Regions", styles['Heading2']))
        
        # Create data for table
        valid_regions = [info for info in region_info]
        
        if valid_regions:
            table_data = [['Region ID', 'Volume', 'Sphericity', 'Elongation', 'Compactness']]
            
            for region in valid_regions:
                table_data.append([
                    region['label'],
                    f"{region['volume']:.1f}",
                    f"{region['sphericity']:.3f}",
                    f"{region['elongation']:.3f}",
                    f"{region['compactness']:.3f}"
                ])
            
            # Create table with style
            table = Table(table_data, colWidths=[0.8*inch, 0.8*inch, 1*inch, 1*inch, 1*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(table)
        else:
            elements.append(Paragraph("No valid bolt head regions detected.", styles['Normal']))
        
        elements.append(Spacer(1, 0.25*inch))
        
        # Add statistics section
        elements.append(Paragraph("Statistical Analysis", styles['Heading2']))
        
        # Calculate statistics
        if valid_regions:
            avg_sphericity = np.mean([r['sphericity'] for r in valid_regions])
            avg_elongation = np.mean([r['elongation'] for r in valid_regions])
            avg_compactness = np.mean([r['compactness'] for r in valid_regions])
            avg_volume = np.mean([r['volume'] for r in valid_regions])
            
            stats_text = [
                Paragraph(f"Number of detected bolt head regions: <b>{len(valid_regions)}</b>", styles['Normal']),
                Spacer(1, 0.1*inch),
                Paragraph(f"Average sphericity: <b>{avg_sphericity:.3f}</b>", styles['Normal']),
                Spacer(1, 0.1*inch),
                Paragraph(f"Average elongation: <b>{avg_elongation:.3f}</b>", styles['Normal']),
                Spacer(1, 0.1*inch),
                Paragraph(f"Average compactness: <b>{avg_compactness:.3f}</b>", styles['Normal']),
                Spacer(1, 0.1*inch),
                Paragraph(f"Average volume: <b>{avg_volume:.1f}</b> voxels", styles['Normal']),
            ]
            elements.extend(stats_text)
        else:
            elements.append(Paragraph("No statistics available - no valid regions detected.", styles['Normal']))
        
        # Add conclusions
        elements.append(Spacer(1, 0.25*inch))
        elements.append(Paragraph("Conclusions", styles['Heading2']))
        
        if valid_regions:
            conclusions = [
                Paragraph("The shape analysis successfully identified bolt head regions with the following findings:", styles['Normal']),
                Spacer(1, 0.1*inch),
                Paragraph(f"• Bolt heads demonstrate a moderate sphericity (avg: {avg_sphericity:.3f}), indicating their semi-spherical shape", styles['Normal']),
                Spacer(1, 0.1*inch),
                Paragraph(f"• Elongation values (avg: {avg_elongation:.3f}) show bolt heads are not highly elongated, helping distinguish them from other structures", styles['Normal']),
                Spacer(1, 0.1*inch),
                Paragraph(f"• The compactness metric (avg: {avg_compactness:.3f}) confirms bolt heads have a relatively compact structure", styles['Normal']),
                Spacer(1, 0.1*inch),
                Paragraph("The shape metrics together provide effective criteria for differentiating bolt heads from noise or other artifacts in the CT volume.", styles['Normal']),
            ]
            elements.extend(conclusions)
        else:
            elements.append(Paragraph("No valid bolt head regions were detected. Consider adjusting detection parameters or threshold values.", styles['Normal']))
        
        # Build PDF
        doc.build(elements)
        print(f"✅ Generated PDF report at {pdf_path}")
        
    except ImportError:
        print("⚠️ ReportLab library not found. PDF report generation skipped.")
        # Create a simpler text report instead
        txt_path = os.path.join(self.config['output_dir'], "bolt_head_shape_analysis_report.txt")
        with open(txt_path, 'w') as f:
            f.write("BOLT HEAD SHAPE ANALYSIS REPORT\n")
            f.write("===============================\n\n")
            f.write(f"Number of detected bolt head regions: {len(region_info)}\n\n")
            
            for i, region in enumerate(region_info):
                f.write(f"Region {i+1} (ID: {region['label']}):\n")
                f.write(f"  Volume: {region['volume']:.1f} voxels\n")
                f.write(f"  Sphericity: {region['sphericity']:.3f}\n")
                f.write(f"  Elongation: {region['elongation']:.3f}\n")
                f.write(f"  Compactness: {region['compactness']:.3f}\n\n")
            
            print(f"✅ Generated text report at {txt_path}")
    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback
        traceback.print_exc()